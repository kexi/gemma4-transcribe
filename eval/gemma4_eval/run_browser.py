"""実 Chrome（WebGPU）で評価ページを動かし、クリップごとの結果を raw JSONL に追記する。

使い方: uv run --project eval python -m gemma4_eval.run_browser --models e2b,e4b --limit 10

- モデルごとに、まずクリップ 0 件（limit=0）のページでモデルを読み込ませてダウンロードを済ませ（first_load_ms）、
  計測するページではキャッシュからの読み込み時間（load_ms）を測る。ダウンロード時間は回線次第で、モデルの比較にならないため。
  そのページで実際にダウンロードしたか（first_load_downloaded）も Cache API のファイル数の増減から記録する
- --run-id は YYYYMMDDTHHMMSSZ の形式で、既存の run（run.json がある）だけを再開できる。空なら新しい run_id
- 計測するページは読み込み後に先頭クリップを 1 回だけ計測せずに流してから測る（warmup_ms。シェーダのコンパイル等を外すため）
- 結果は届いたものから 1 行ずつ追記・fsync する。途中で落ちてもそこまでの結果は残り、
  同じ --run-id で再実行すると記録済みのクリップを飛ばして続きから測る（offset / limit / 言語は run.json の値を使う）
- 1 クリップが --clip-timeout を超えたら、そのクリップをタイムアウトとして記録し、
  ページを開き直して次のクリップから続ける（Worker が固まったままでは後続も進まないため）
- ページがクラッシュしたら、処理中だったクリップ（window.__eval.currentId）から開き直して測り直し、
  同じクリップで 2 回クラッシュしたときだけ失敗として記録する
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from .just_args import reassign_named
from .log import log
from .paths import CHROME_PROFILE_DIR, DIST_EVAL_DIR, MANIFEST_PATH, RAW_RESULTS_DIR, REPO_ROOT
from .run_id import is_valid_run_id, new_run_id
from .server import DATA_PREFIX, start_server

# 評価ページの Cache API（モデルのキャッシュ）はオリジン単位なので、ポートが毎回変わると
# 永続プロファイルを使っていても数 GB を毎回ダウンロードし直すことになる。既定は固定ポートにする
DEFAULT_PORT = 8765
DEFAULT_LOAD_TIMEOUT_S = 40 * 60
DEFAULT_CLIP_TIMEOUT_S = 5 * 60
POLL_INTERVAL_S = 1.0
HEARTBEAT_INTERVAL_S = 30.0
# 進捗の無いままページを開き直す回数の上限。同じ原因で落ち続けるときに無限に繰り返さないため
MAX_RESTARTS_WITHOUT_PROGRESS = 2
# 同じクリップの処理中にページがこの回数クラッシュしたら、そのクリップを失敗として記録する。
# Why not 1 回目で記録: クラッシュの直前に終わった結果をまだ読み取っていないことがあり、
# 1 回だけでは「そのクリップが原因」とは言い切れないため、開き直して同じクリップでもう一度落ちるかを見る
CRASHES_BEFORE_FAILURE = 2
# 既定ポートが塞がっていたときに順に試す個数（DEFAULT_PORT を含む）
PORT_ATTEMPTS = 10
DEFAULT_MODEL = "e2b"
# 再開時に run.json の値を使う設定と、新規実行での既定値
DEFAULT_SETTINGS: dict[str, Any] = {"offset": 0, "limit": None, "language": "Japanese"}

# results は前回までに読んだ件数より後ろだけを返す。数百件の結果を毎秒すべて転送しないため
_SNAPSHOT_JS = """
(seen) => {
  const state = window.__eval;
  if (state === undefined || state === null) return null;
  return {
    phase: state.phase,
    model: state.model,
    modelInfo: state.modelInfo ?? null,
    loadMs: state.loadMs,
    warmupMs: state.warmupMs ?? null,
    currentId: state.currentId ?? null,
    cachedModelFiles: state.cachedModelFiles ?? null,
    total: state.total,
    completed: state.completed,
    error: state.error,
    results: state.results.slice(seen),
  };
}
"""

_PROBE_JS = """
async () => {
  const brands = navigator.userAgentData
    ? await navigator.userAgentData.getHighEntropyValues(['fullVersionList', 'platformVersion'])
    : null;
  const adapter = 'gpu' in navigator ? await navigator.gpu.requestAdapter() : null;
  const info = adapter?.info;
  return {
    userAgent: navigator.userAgent,
    fullVersionList: brands?.fullVersionList ?? null,
    platform: brands?.platform ?? null,
    platformVersion: brands?.platformVersion ?? null,
    gpu: info
      ? {
          vendor: info.vendor,
          architecture: info.architecture,
          device: info.device,
          description: info.description,
        }
      : null,
  };
}
"""

_CAMEL_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")


class UnplannedClipError(Exception):
    """評価ページが、この実行で予定していない ID の結果を返した（manifest や範囲の取り違え）。"""


def to_snake_case(name: str) -> str:
    """durationS → duration_s。EvalResult のキーを raw JSONL の snake_case に揃える。"""
    return _CAMEL_BOUNDARY.sub("_", name).lower()


def raw_row(
    result: dict, model: str, run_id: str, load_ms: float | None, warmup_ms: float | None
) -> dict:
    """EvalResult（camelCase）に model / run_id / load_ms / warmup_ms を足して snake_case にした 1 行。"""
    row = {to_snake_case(key): value for key, value in result.items()}
    row.update({"model": model, "run_id": run_id, "load_ms": load_ms, "warmup_ms": warmup_ms})
    return row


def plan_items(items: Sequence[dict], offset: int, limit: int | None) -> list[dict]:
    """評価ページの selectItems と同じ切り出し（offset から limit 件）。"""
    end = None if limit is None else offset + limit
    return list(items[offset:end])


def planned_ids_digest(planned: Sequence[dict]) -> str:
    """予定したクリップ ID の並びの sha256。再開時に manifest が作り直されていないかを確かめるため run.json に残す。"""
    joined = "\n".join(item["id"] for item in planned)
    return hashlib.sha256(joined.encode()).hexdigest()


def planned_datasets(planned: Sequence[dict]) -> dict[str, dict]:
    """データセットごとの予定件数と音声長の合計（manifest の値）。集計 JSON の datasets に使う。"""
    datasets: dict[str, dict] = {}
    for item in planned:
        entry = datasets.setdefault(item["dataset"], {"n": 0, "audio_s": 0.0})
        entry["n"] += 1
        entry["audio_s"] = round(entry["audio_s"] + item["duration_s"], 3)
    return datasets


def resolve_settings(existing: dict | None, requested: dict[str, Any]) -> dict[str, Any]:
    """offset / limit / language を決める。requested の None は「未指定」。

    既存の run.json があれば（--run-id で再開）その値を使い、明示された値が食い違えば ValueError にする。
    Why not 黙って新しい値を使う: 同じ run_id の中で範囲や言語が混ざると、記録済みの行と新しい行が
    別条件の結果になり、集計が何を測ったものか分からなくなるため
    """
    resolved: dict[str, Any] = {}
    mismatches = []
    for name, default in DEFAULT_SETTINGS.items():
        value = requested.get(name)
        if existing is None:
            resolved[name] = default if value is None else value
            continue
        recorded = existing.get(name, default)
        is_mismatch = value is not None and value != recorded
        if is_mismatch:
            mismatches.append(f"--{name} {value}（run.json は {recorded}）")
        resolved[name] = recorded
    if mismatches:
        raise ValueError(
            "再開する run と条件が違います: "
            + "、".join(mismatches)
            + "。指定を外すか、新しい run_id で実行してください"
        )
    return resolved


def read_recorded_ids(path: Path) -> set[str]:
    """既存の raw JSONL に記録済みのクリップ ID。最後の行が書きかけで壊れていても読める分だけ返す。"""
    if not path.exists():
        return set()
    recorded = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            recorded.add(json.loads(line)["id"])
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return recorded


def append_jsonl(path: Path, row: dict) -> None:
    # 1 行ごとに fsync する。Chrome やこのプロセスが落ちても、書き終えた行は失われない
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def eval_url(
    base: str,
    model: str,
    offset: int,
    limit: int | None,
    language: str,
    manifest_name: str = "manifest.json",
) -> str:
    """評価ページの URL。manifest は /data/ 配下（サーバは --manifest のディレクトリを /data/ で返す）。"""
    params = {
        "model": model,
        "manifest": f"{DATA_PREFIX}/{manifest_name}",
        "offset": offset,
        "language": language,
    }
    if limit is not None:
        params["limit"] = limit
    return f"{base}/eval.html?{urlencode(params)}"


def git_info() -> dict:
    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return completed.stdout.strip()

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def _sysctl(name: str) -> str | None:
    try:
        completed = subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def host_info() -> dict:
    """OS と機種（集計 JSON の environment.platform / machine）。

    Why not ブラウザの userAgentData.platformVersion: macOS では版が丸められたり古いままのことがあり、
    OS の更新で速度が変わったかを追えないため、ホスト側で取る
    """
    is_macos = platform.system() == "Darwin"
    if not is_macos:
        return {"platform": platform.platform(), "machine": platform.machine() or None}
    # hw.model は "Mac14,5" のような機種 ID（シリアル番号ではない）
    parts = [_sysctl("hw.model"), _sysctl("machdep.cpu.brand_string")]
    return {
        "platform": f"macOS {platform.mac_ver()[0]} ({platform.machine()})",
        "machine": " / ".join(part for part in parts if part) or None,
    }


def chrome_version(probe: dict) -> str | None:
    """fullVersionList から Chrome の完全な版を取り出す（UA 文字列は版が丸められているため）。"""
    for brand in probe.get("fullVersionList") or []:
        is_chrome = brand.get("brand") in ("Google Chrome", "Chromium")
        if is_chrome:
            return brand.get("version")
    return None


@dataclass
class SessionOutcome:
    """1 回ページを開いて評価した結果。"""

    # done / load-error / load-timeout / page-error / clip-timeout / crashed / unplanned-clip
    status: str
    load_ms: float | None = None
    warmup_ms: float | None = None
    new_results: int = 0
    message: str | None = None
    # clip-timeout / crashed のとき、最後の進捗から止まるまでの時間
    stalled_ms: float | None = None
    # 最後に読み取ったときに処理中だったクリップ（ウォームアップを含む）
    current_id: str | None = None
    model_info: dict | None = None
    # Cache API にあるこのモデルのファイル数（読み込みの直前 / 直後）。数えられなければ None
    cached_files_before: int | None = None
    cached_files_after: int | None = None


@dataclass
class ModelRun:
    model: str
    planned: list[dict]
    out_path: Path
    run_id: str
    recorded: set[str] = field(default_factory=set)
    # クリップ ID → そのクリップの処理中にページがクラッシュした回数
    crash_counts: Counter[str] = field(default_factory=Counter)

    def __post_init__(self) -> None:
        self._planned_by_id = {item["id"]: item for item in self.planned}

    def pending_index(self) -> int | None:
        for index, item in enumerate(self.planned):
            is_pending = item["id"] not in self.recorded
            if is_pending:
                return index
        return None

    def record(self, row: dict) -> bool:
        """1 行を追記する。記録済みなら何もしない。予定外の ID なら UnplannedClipError。

        予定外の ID を黙って書くと、別の manifest や範囲の結果が同じ run に混ざるため、書かずに止める。
        """
        is_planned = row["id"] in self._planned_by_id
        if not is_planned:
            raise UnplannedClipError(f"予定していないクリップの結果が届きました: {row['id']}")
        is_duplicate = row["id"] in self.recorded
        if is_duplicate:
            return False
        append_jsonl(self.out_path, row)
        self.recorded.add(row["id"])
        return True

    def stuck_row_for(self, outcome: SessionOutcome) -> dict | None:
        """セッションがクリップの途中で止まったとき、失敗として記録する行。記録しないなら None。

        - clip-timeout: ページは生きていて、最後のポーリングで読んだ currentId が止まったクリップなので、すぐ記録する
        - crashed: currentId は最後のポーリング時点の値で、その後に終わった結果は読めていないことがある。
          同じクリップで CRASHES_BEFORE_FAILURE 回落ちたときだけ記録し、それまでは開き直して測り直す
        """
        clip_id = outcome.current_id
        has_loaded = outcome.load_ms is not None
        is_unknown_clip = clip_id is None or clip_id not in self._planned_by_id
        if not has_loaded or is_unknown_clip or clip_id in self.recorded:
            return None
        elapsed_ms = outcome.stalled_ms or 0.0
        if outcome.status == "clip-timeout":
            error = f"timeout: {elapsed_ms / 1000:.0f} 秒間結果が出ませんでした"
            return stuck_row(self._planned_by_id[clip_id], self, outcome, elapsed_ms, error)
        if outcome.status != "crashed":
            return None
        self.crash_counts[clip_id] += 1
        is_repeated = self.crash_counts[clip_id] >= CRASHES_BEFORE_FAILURE
        if not is_repeated:
            return None
        error = f"crashed: {self.crash_counts[clip_id]} 回クラッシュしました（{outcome.message}）"
        return stuck_row(self._planned_by_id[clip_id], self, outcome, elapsed_ms, error)


def stuck_row(
    item: dict, model_run: ModelRun, outcome: SessionOutcome, elapsed_ms: float, error: str
) -> dict:
    """ページ側で結果が出なかった（タイムアウト・クラッシュ）クリップの代わりに記録する行。

    黙って欠落させず、失敗として集計に出すため。
    """
    return raw_row(
        {
            "id": item["id"],
            "dataset": item["dataset"],
            "reference": item["reference"],
            "hypothesis": "",
            # デコード後の実長は分からないので manifest の値で埋める（失敗クリップは RTF に使わない）
            "durationS": item["duration_s"],
            "decodeMs": 0,
            "inferMs": elapsed_ms,
            "segments": 0,
            "error": error,
        },
        model_run.model,
        model_run.run_id,
        outcome.load_ms,
        outcome.warmup_ms,
    )


def run_session(
    context,
    url: str,
    model_run: ModelRun,
    load_timeout_s: float,
    clip_timeout_s: float,
    now: Callable[[], float] = time.monotonic,
) -> SessionOutcome:
    """評価ページを 1 回開き、done / error / タイムアウトまで window.__eval をポーリングする。"""
    from playwright.sync_api import Error as PlaywrightError

    page = context.new_page()
    crashed = {"value": False}
    page.on("crash", lambda: crashed.update(value=True))
    page.on(
        "console",
        lambda message: (
            log("page_console", model=model_run.model, level=message.type, text=message.text)
            if message.type in ("error", "warning")
            else None
        ),
    )
    page.on("pageerror", lambda error: log("page_error", model=model_run.model, message=str(error)))

    outcome = SessionOutcome(status="crashed")
    seen = 0
    started_at = now()
    last_progress_at = started_at
    last_heartbeat_at = started_at
    is_loaded = False
    is_warmed = False
    try:
        page.goto(url, wait_until="domcontentloaded")
        while True:
            time.sleep(POLL_INTERVAL_S)
            if crashed["value"]:
                outcome.status, outcome.message = "crashed", "ページがクラッシュしました"
                outcome.stalled_ms = (now() - last_progress_at) * 1000
                return outcome
            snapshot = page.evaluate(_SNAPSHOT_JS, seen)
            current = now()
            if snapshot is None:
                is_load_timeout = current - started_at > load_timeout_s
                if is_load_timeout:
                    outcome.status = "load-timeout"
                    return outcome
                continue

            outcome.load_ms = snapshot["loadMs"]
            outcome.warmup_ms = snapshot["warmupMs"]
            outcome.current_id = snapshot["currentId"]
            outcome.model_info = snapshot["modelInfo"] or outcome.model_info
            cached = snapshot["cachedModelFiles"] or {}
            outcome.cached_files_before = cached.get("before")
            outcome.cached_files_after = cached.get("after")
            # phase ではなく loadMs で判定する。読み込みから完了までが 1 回のポーリング間隔に収まると
            # running を一度も観測しないまま done になるため
            has_loaded_now = snapshot["loadMs"] is not None and not is_loaded
            if has_loaded_now:
                is_loaded = True
                # クリップ（ウォームアップを含む）のタイムアウトはモデルの読み込みが終わった時点から数える
                last_progress_at = current
                log(
                    "model_loaded",
                    run_id=model_run.run_id,
                    model=model_run.model,
                    load_ms=snapshot["loadMs"],
                )
            has_warmed_now = snapshot["warmupMs"] is not None and not is_warmed
            if has_warmed_now:
                is_warmed = True
                last_progress_at = current
                log(
                    "warmup_done",
                    run_id=model_run.run_id,
                    model=model_run.model,
                    warmup_ms=snapshot["warmupMs"],
                )

            for result in snapshot["results"]:
                seen += 1
                row = raw_row(
                    result,
                    model_run.model,
                    model_run.run_id,
                    snapshot["loadMs"],
                    snapshot["warmupMs"],
                )
                is_new = model_run.record(row)
                if not is_new:
                    continue
                outcome.new_results += 1
                last_progress_at = current
                log(
                    "clip_done",
                    run_id=model_run.run_id,
                    model=model_run.model,
                    id=row["id"],
                    recorded=len(model_run.recorded),
                    planned=len(model_run.planned),
                    duration_s=row["duration_s"],
                    infer_ms=round(row["infer_ms"]),
                    error=row["error"],
                )

            phase = snapshot["phase"]
            if phase == "done":
                outcome.status = "done"
                return outcome
            if phase == "error":
                is_load_failure = snapshot["loadMs"] is None
                outcome.status = "load-error" if is_load_failure else "page-error"
                outcome.message = snapshot["error"]
                return outcome

            is_loading = phase == "loading-model"
            is_load_timeout = is_loading and current - started_at > load_timeout_s
            if is_load_timeout:
                outcome.status = "load-timeout"
                return outcome
            is_on_clip = phase in ("warming-up", "running")
            is_clip_timeout = is_on_clip and current - last_progress_at > clip_timeout_s
            if is_clip_timeout:
                outcome.status = "clip-timeout"
                outcome.stalled_ms = (current - last_progress_at) * 1000
                return outcome

            is_heartbeat_due = current - last_heartbeat_at >= HEARTBEAT_INTERVAL_S
            if is_heartbeat_due:
                last_heartbeat_at = current
                log(
                    "heartbeat",
                    run_id=model_run.run_id,
                    model=model_run.model,
                    phase=phase,
                    current_id=snapshot["currentId"],
                    completed=snapshot["completed"],
                    total=snapshot["total"],
                    elapsed_s=round(current - started_at),
                )
    except UnplannedClipError as error:
        outcome.status, outcome.message = "unplanned-clip", str(error)
        return outcome
    except PlaywrightError as error:
        outcome.status, outcome.message = "crashed", str(error)
        outcome.stalled_ms = (now() - last_progress_at) * 1000
        return outcome
    finally:
        # クラッシュしたページは close も失敗しうる。後片付けの失敗で評価全体を止めない
        with contextlib.suppress(PlaywrightError):
            page.close()


@dataclass
class ModelOutcome:
    status: str
    # ダウンロード専用セッションでの読み込み時間（first_load_downloaded が True のときだけダウンロードを含む）
    first_load_ms: float | None = None
    # そのセッションでモデルのファイルをダウンロードしたか。None は不明
    first_load_downloaded: bool | None = None
    model_info: dict | None = None


def detect_download(before: int | None, after: int | None) -> bool | None:
    """読み込みの直前・直後に数えた Cache API 上のモデルのファイル数から、ダウンロードが起きたかを決める。

    - 増えた: 足りないファイルを取得して Cache API に保存した → True
    - 0 より多いまま変わらない: すべてキャッシュから読んだ → False
    - 読み込み後も 0、または数えられない: Cache API が使われていない（ブロック・容量不足など）ので判断できない → None
    Why not Transformers.js の進捗（load-progress の loadedBytes）で判断: キャッシュから読むときも
    同じ progress イベントがバイト数付きで届くため、ダウンロードとキャッシュ読み込みを区別できない。
    既知の限界: 一部のファイルだけ保存に失敗した場合は、ダウンロードしても False になりうる
    """
    is_unknown = before is None or after is None or after == 0
    if is_unknown:
        return None
    return after > before


def prefetch_download(attempts: Sequence[SessionOutcome]) -> bool | None:
    """ダウンロード専用セッション（クラッシュで開き直した分を含む）全体でダウンロードが起きたか。

    最初の試行の読み込み直前と、最後の試行の読み込み直後を比べる。
    Why not 最後の試行だけで判断: 1 回目がダウンロード途中で落ちると、保存済みの分は 2 回目にはキャッシュから読まれ、
    残りが少なければ「ほぼキャッシュ」の読み込みでも、1 回目のダウンロードを見落とすことになるため
    """
    if not attempts:
        return None
    return detect_download(attempts[0].cached_files_before, attempts[-1].cached_files_after)


@dataclass
class SessionSettings:
    base_url: str
    offset: int
    language: str
    manifest_name: str
    load_timeout_s: float
    clip_timeout_s: float


def prefetch_model(
    context, model_run: ModelRun, settings: SessionSettings
) -> tuple[SessionOutcome, bool | None]:
    """クリップ 0 件（limit=0）でページを開き、モデルの読み込み（とダウンロード）だけを済ませる。

    計測するセッションの load_ms をキャッシュからの読み込み時間にするため。
    最後の試行の結果と、全試行を通してダウンロードが起きたか（prefetch_download）を返す。
    Why not 計測セッションの load_ms からダウンロード分を差し引く: Transformers.js の進捗はファイル単位で、
    ダウンロードとセッション作成（WebGPU の初期化）が重なって進むため、きれいに分けられない
    """
    url = eval_url(
        settings.base_url, model_run.model, 0, 0, settings.language, settings.manifest_name
    )
    attempts: list[SessionOutcome] = []
    for attempt in range(1, MAX_RESTARTS_WITHOUT_PROGRESS + 2):
        log("prefetch_start", run_id=model_run.run_id, model=model_run.model, attempt=attempt)
        outcome = run_session(
            context, url, model_run, settings.load_timeout_s, settings.clip_timeout_s
        )
        attempts.append(outcome)
        log(
            "prefetch_end",
            run_id=model_run.run_id,
            model=model_run.model,
            status=outcome.status,
            load_ms=outcome.load_ms,
            cached_files_before=outcome.cached_files_before,
            cached_files_after=outcome.cached_files_after,
            message=outcome.message,
        )
        # クラッシュ（ダウンロード途中のタブ落ちなど）だけは開き直す。キャッシュ済みの分は再取得されない
        is_retryable = outcome.status == "crashed"
        if not is_retryable:
            break
    return attempts[-1], prefetch_download(attempts)


def run_model(context, model_run: ModelRun, settings: SessionSettings) -> ModelOutcome:
    """1 モデル分を最後まで（または諦めるまで）流し、最終状態を返す。"""
    if model_run.pending_index() is None:
        return ModelOutcome(status="done")

    prefetched, downloaded = prefetch_model(context, model_run, settings)
    result = ModelOutcome(
        status=prefetched.status,
        first_load_ms=prefetched.load_ms,
        first_load_downloaded=downloaded,
        model_info=prefetched.model_info,
    )
    if prefetched.status != "done":
        result.status = f"prefetch-failed: {prefetched.status}"
        return result

    restarts_without_progress = 0
    while True:
        pending = model_run.pending_index()
        if pending is None:
            result.status = "done"
            return result
        remaining = len(model_run.planned) - pending
        url = eval_url(
            settings.base_url,
            model_run.model,
            settings.offset + pending,
            remaining,
            settings.language,
            settings.manifest_name,
        )
        log("session_start", run_id=model_run.run_id, model=model_run.model, url=url)
        session_started_at = time.monotonic()
        outcome = run_session(
            context, url, model_run, settings.load_timeout_s, settings.clip_timeout_s
        )
        result.model_info = outcome.model_info or result.model_info
        log(
            "session_end",
            run_id=model_run.run_id,
            model=model_run.model,
            status=outcome.status,
            new_results=outcome.new_results,
            load_ms=outcome.load_ms,
            warmup_ms=outcome.warmup_ms,
            current_id=outcome.current_id,
            message=outcome.message,
            elapsed_s=round(time.monotonic() - session_started_at),
        )

        is_unrecoverable = outcome.status in ("load-error", "load-timeout", "unplanned-clip")
        if is_unrecoverable:
            result.status = outcome.status
            return result
        if outcome.status == "done":
            # limit で切った範囲より manifest が短いなど、ページ側の total が少ない場合はここで終わる
            result.status = "done"
            return result

        # クリップ途中で止まった（タイムアウト・クラッシュ）なら、処理中だったクリップを失敗として記録して先へ進む。
        # page-error は Worker が落ちたクリップの結果をページ側が記録済みなので、そのまま次から再開する
        stuck = model_run.stuck_row_for(outcome)
        if stuck is not None:
            model_run.record(stuck)
            outcome.new_results += 1
            log(
                "clip_failed",
                run_id=model_run.run_id,
                model=model_run.model,
                id=stuck["id"],
                error=stuck["error"],
            )

        made_progress = outcome.new_results > 0
        restarts_without_progress = 0 if made_progress else restarts_without_progress + 1
        is_giving_up = restarts_without_progress > MAX_RESTARTS_WITHOUT_PROGRESS
        if is_giving_up:
            result.status = f"gave-up: {outcome.status}"
            return result


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--models",
        default=DEFAULT_MODEL,
        help="models.ts の key をカンマ区切りで（例: e2b,e2b-qat,e4b,e4b-qat）。1 つの run_id で順に測る",
    )
    parser.add_argument(
        "--limit",
        default="",
        help="各モデルで評価する件数の上限（空なら全件。再開時は空なら run.json の値）",
    )
    parser.add_argument(
        "--offset",
        default="",
        help="manifest の何件目から測るか（空なら 0。再開時は空なら run.json の値）",
    )
    parser.add_argument("--language", default="", help="空なら Japanese（再開時は run.json の値）")
    parser.add_argument(
        "--run-id",
        default="",
        help="既存の run_id（YYYYMMDDTHHMMSSZ、run.json があるもの）を渡すと記録済みのクリップを飛ばして続きから測る（空なら新しい run_id）",
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--dist-dir", type=Path, default=DIST_EVAL_DIR)
    parser.add_argument("--profile-dir", type=Path, default=CHROME_PROFILE_DIR)
    parser.add_argument("--raw-dir", type=Path, default=RAW_RESULTS_DIR)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--load-timeout", type=float, default=DEFAULT_LOAD_TIMEOUT_S, help="秒")
    parser.add_argument("--clip-timeout", type=float, default=DEFAULT_CLIP_TIMEOUT_S, help="秒")
    return parser.parse_args(argv)


def start_server_near(
    dist_dir: Path, data_dir: Path, port: int, attempts: int = PORT_ATTEMPTS
) -> ThreadingHTTPServer:
    """port から順に attempts 個のポートを試してサーバを起動する。全部塞がっていれば OS に選ばせる。

    Why not すぐに空きポート（port=0）へ逃げる: 毎回違うポートになり、オリジン単位のモデルキャッシュが
    使えず数 GB を毎回ダウンロードし直すため。隣のポートなら、同じ衝突が続く限り次回も同じオリジンになる
    """
    for candidate in range(port, port + attempts):
        try:
            server = start_server(dist_dir, data_dir, port=candidate)
        except OSError as error:
            log("port_unavailable", port=candidate, message=str(error))
            continue
        return server
    # ポートが塞がっていても評価自体はできる。ただしモデルのキャッシュを使えず再ダウンロードになる
    log("port_fallback", fallback="ephemeral", tried=[port, port + attempts - 1])
    return start_server(dist_dir, data_dir, port=0)


def parse_count(name: str, raw: str) -> int | None:
    """空文字は未指定（just の `limit=""` がそのまま渡ってくるため）。"""
    is_unset = raw.strip() == ""
    if is_unset:
        return None
    value = int(raw)
    if value < 0:
        raise ValueError(f"--{name} は 0 以上: {raw}")
    return value


def parse_limit(raw: str) -> int | None:
    return parse_count("limit", raw)


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_run_id(requested: str, raw_dir: Path, now: datetime | None = None) -> str:
    """--run-id を検証して使う run_id を返す。空なら新しい run_id。

    形式が違う、または run.json の無い run_id を明示されたら ValueError。
    Why not 存在しない run_id を新しい run として始める: `run=` の打ち間違いが黙って別の run になり、
    再開したつもりの結果が分かれて記録されるうえ、名前順で最新になれば latest.json まで奪うため
    """
    run_id = requested.strip()
    if run_id == "":
        return new_run_id(now)
    if not is_valid_run_id(run_id):
        raise ValueError(f"--run-id は YYYYMMDDTHHMMSSZ の形式です: {run_id}")
    has_run = (raw_dir / run_id / "run.json").exists()
    if not has_run:
        raise ValueError(
            f"再開する run が見つかりません: {raw_dir / run_id / 'run.json'}。"
            "新しく測るときは run を空にしてください"
        )
    return run_id


def record_chrome(metadata: dict, chrome: dict, is_resume: bool) -> None:
    """ブラウザの情報を run.json に書く。最初の実行の chrome は上書きせず、再開時の値は resumes の最後に積む。

    Why not 再開のたびに chrome を書き換える: 記録済みの行の多くは最初の実行の Chrome / GPU で測ったもので、
    集計の実行環境が途中で別の版に入れ替わると、どの環境の結果かを誤って伝えるため
    """
    has_original = metadata.get("chrome") is not None
    if not has_original:
        metadata["chrome"] = chrome
    resumes = metadata.get("resumes") or []
    if is_resume and resumes:
        resumes[-1]["chrome"] = chrome


def record_first_load(model_metadata: dict, outcome: ModelOutcome) -> None:
    """最初に測れた初回読み込み時間を、そのときダウンロードしたかと組で残す。

    再開時の読み込みはキャッシュからになるので、既に値があれば上書きしない。
    時間とダウンロードの有無は同じセッションの値でないと意味が無いので、必ず一緒に書く
    """
    has_first_load = model_metadata.get("first_load_ms") is not None
    if has_first_load:
        return
    model_metadata["first_load_ms"] = outcome.first_load_ms
    model_metadata["first_load_downloaded"] = outcome.first_load_downloaded


def _browser_metadata(context, base_url: str, manifest_name: str) -> dict:
    probe_page = context.new_page()
    try:
        # 評価ページと同じオリジン（secure context）で UA と GPU の情報を取る。
        # eval.html を使わないのは、モデルの読み込みを始めさせないため
        probe_page.goto(f"{base_url}{DATA_PREFIX}/{manifest_name}", wait_until="domcontentloaded")
        probe = probe_page.evaluate(_PROBE_JS)
    finally:
        probe_page.close()
    return {
        "version": chrome_version(probe),
        "user_agent": probe["userAgent"],
        "platform": probe["platform"],
        "platform_version": probe["platformVersion"],
        "gpu": probe["gpu"],
    }


def _new_metadata(
    run_id: str,
    started_at: str,
    settings: dict,
    manifest_path: Path,
    manifest: dict,
    planned: list[dict],
) -> dict:
    return {
        "run_id": run_id,
        "started_at": started_at,
        "git": git_info(),
        **settings,
        "manifest": {
            "path": manifest_path.name,
            "items": len(manifest["items"]),
            "per_dataset": manifest.get("per_dataset"),
            "excluded_over_30s": manifest.get("excluded_over_30s"),
            "created_from": manifest.get("created_from"),
        },
        "planned": {
            "items": len(planned),
            "ids_sha256": planned_ids_digest(planned),
            "datasets": planned_datasets(planned),
        },
        "models": {},
        "host": host_info(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        given = reassign_named({"models": args.models, "limit": args.limit, "run": args.run_id})
    except ValueError as error:
        log("invalid_arguments", message=str(error))
        return 2
    models = [
        name.strip() for name in (given["models"] or DEFAULT_MODEL).split(",") if name.strip()
    ]
    try:
        run_id = resolve_run_id(given["run"], args.raw_dir)
    except ValueError as error:
        log(
            "invalid_arguments",
            message=str(error),
            hint="既存の run_id は eval/results/raw/ を確認",
        )
        return 2

    has_page = (args.dist_dir / "eval.html").exists()
    if not has_page:
        log("missing_input", path=str(args.dist_dir / "eval.html"), hint="just eval-build")
        return 1
    if not args.manifest.exists():
        log("missing_input", path=str(args.manifest), hint="just eval-prepare")
        return 1

    run_dir = args.raw_dir / run_id
    metadata_path = run_dir / "run.json"
    existing = _read_json(metadata_path)
    try:
        settings = resolve_settings(
            existing,
            {
                "offset": parse_count("offset", args.offset),
                "limit": parse_count("limit", given["limit"]),
                "language": args.language.strip() or None,
            },
        )
    except ValueError as error:
        log("invalid_arguments", run_id=run_id, message=str(error))
        return 2

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    planned = plan_items(manifest["items"], settings["offset"], settings["limit"])
    recorded_digest = ((existing or {}).get("planned") or {}).get("ids_sha256")
    is_manifest_changed = recorded_digest is not None and recorded_digest != planned_ids_digest(
        planned
    )
    if is_manifest_changed:
        # 作り直した manifest で続きを測ると、記録済みの行と別のクリップ集合が 1 つの run に混ざる
        log("manifest_changed", run_id=run_id, hint="新しい run_id で実行してください")
        return 2
    run_dir.mkdir(parents=True, exist_ok=True)

    from playwright.sync_api import sync_playwright

    server = start_server_near(args.dist_dir, args.manifest.parent, args.port)
    # localhost / 127.0.0.1 は http でも secure context なので WebGPU が使える
    base_url = f"http://127.0.0.1:{server.server_port}"

    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    if existing is None:
        metadata = _new_metadata(run_id, started_at, settings, args.manifest, manifest, planned)
    else:
        metadata = existing
        # 最初の実行の条件（git commit など）は残し、再開した時点の状態は別に積む
        metadata.setdefault("resumes", []).append({"started_at": started_at, "git": git_info()})
        metadata.setdefault("models", {})
    exit_code = 0
    log(
        "run_start",
        run_id=run_id,
        models=models,
        planned=len(planned),
        base_url=base_url,
        resumed=existing is not None,
        **settings,
    )
    session_settings = SessionSettings(
        base_url=base_url,
        offset=settings["offset"],
        language=settings["language"],
        manifest_name=args.manifest.name,
        load_timeout_s=args.load_timeout,
        clip_timeout_s=args.clip_timeout,
    )

    try:
        with sync_playwright() as playwright:
            # channel=chrome: Playwright 同梱の Chromium ではなくインストール済みの Google Chrome を使う。
            # 同梱版は WebGPU の GPU ブロックリストやドライバの扱いが利用者の環境と違いうるため
            # headless=False: headless では WebGPU のアダプタが得られない／ソフトウェア描画になる環境がある
            context = playwright.chromium.launch_persistent_context(
                str(args.profile_dir),
                channel="chrome",
                headless=False,
                no_viewport=True,
            )
            try:
                chrome = _browser_metadata(context, base_url, args.manifest.name)
                record_chrome(metadata, chrome, is_resume=existing is not None)
                write_json(metadata_path, metadata)
                log("browser", run_id=run_id, **chrome)

                for model in models:
                    out_path = run_dir / f"{model}.jsonl"
                    model_run = ModelRun(
                        model=model,
                        planned=planned,
                        out_path=out_path,
                        run_id=run_id,
                        recorded=read_recorded_ids(out_path),
                    )
                    outcome = run_model(context, model_run, session_settings)
                    model_metadata = metadata["models"].setdefault(model, {})
                    model_metadata.update(
                        {
                            "planned": len(planned),
                            "recorded": len(model_run.recorded),
                            "status": outcome.status,
                        }
                    )
                    record_first_load(model_metadata, outcome)
                    if outcome.model_info is not None:
                        model_metadata["info"] = outcome.model_info
                    write_json(metadata_path, metadata)
                    log("model_end", run_id=run_id, model=model, status=outcome.status)
                    is_failed = outcome.status != "done"
                    if is_failed:
                        exit_code = 1
            finally:
                context.close()
    except KeyboardInterrupt:
        log("interrupted", run_id=run_id)
        exit_code = 130
    finally:
        server.shutdown()
        metadata["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        write_json(metadata_path, metadata)

    log("run_end", run_id=run_id, raw_dir=str(run_dir), exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
