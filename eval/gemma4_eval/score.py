"""raw 結果（eval/results/raw/<run_id>/*.jsonl）を集計し、Markdown と JSON に書き出す。

- eval/results/<run_id>.md: 数値と ID だけ（コミット可）。参照・推論テキストは載せない
- eval/results/<run_id>.json と eval/results/latest.json: ベンチマークページ（benchmarks.html）が読む集計 JSON。
  数値・ID・実行環境だけ（コミット可）。latest.json はこれまでに集計した中で最新の run を指す
- eval/results/raw/<run_id>/summary.json: 機械可読の詳細な集計（raw と同じくコミットしない）

1 つの run_id に複数モデルの raw があれば（`just eval-run models="e2b,e4b"`）、全モデルをまとめて集計する。

使い方: uv run --project eval python -m gemma4_eval.score [--run-id <run_id>]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Collection, Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from .just_args import reassign_named
from .log import log
from .metrics import METRIC_NAMES, EditCount, edit_count, speed
from .paths import RAW_RESULTS_DIR, RESULTS_DIR
from .prepare import DATASETS
from .run_id import is_valid_run_id, run_ids

WORST_EXAMPLES = 3
OVERALL = "overall"
# ベンチマークページとの契約の版。フィールドを足すだけなら上げない（ページは知らないキーを無視する）。
# 既存のキーの意味や形を変えるときだけ上げ、ページ側も合わせて直す
SCHEMA_VERSION = 1
# クリップ単体の cer_norm がこれを超えたら外れ値として数える（感度分析用）。
# 100% 超は「参照より多く間違えた」状態で、ReazonSpeech では参照が音声の一部しか書き起こしていないクリップで起きる。
# 外れ値かどうかはモデルごとではなく run 全体でクリップ単位に決める（run_outlier_ids）
OUTLIER_CER_NORM = 1.0
LATEST_NAME = "latest"


def load_rows(run_dir: Path) -> dict[str, list[dict]]:
    """model → 行のリスト。同じ ID が複数あれば最初の行を使う（run_browser は重複を書かないが念のため）。"""
    rows_by_model: dict[str, list[dict]] = {}
    for path in sorted(run_dir.glob("*.jsonl")):
        seen: set[str] = set()
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            is_blank = not line.strip()
            if is_blank:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # 書き込み中に落ちた最終行だけが壊れうる。そこまでの結果は使う
                log("skip_broken_line", path=str(path))
                continue
            is_duplicate = row["id"] in seen
            if is_duplicate:
                continue
            seen.add(row["id"])
            rows.append(row)
        rows_by_model[path.stem] = rows
    return rows_by_model


def _is_failed(row: dict) -> bool:
    return row.get("error") is not None


def _edit_count_json(count: EditCount) -> dict:
    return {"rate": count.rate, "edits": count.edits, "reference_chars": count.reference_chars}


def _sum_counts(counts: Iterable[EditCount]) -> EditCount:
    total = EditCount(0, 0)
    for count in counts:
        total = total + count
    return total


def _norm_counts(rows: Iterable[dict]) -> list[tuple[str, EditCount]]:
    """成功したクリップの (ID, cer_norm の編集数)。失敗クリップは CER に使わないので含めない。"""
    return [
        (row["id"], edit_count(row["reference"], row["hypothesis"], "cer_norm"))
        for row in rows
        if not _is_failed(row)
    ]


def run_outlier_ids(rows_by_model: dict[str, list[dict]]) -> list[str]:
    """run 内のどれか 1 つのモデルでもクリップ単体の cer_norm が OUTLIER_CER_NORM を超えたクリップの ID（名前順）。

    感度分析では、この同じ集合を全モデルから除く。
    Why not モデルごとに外れ値を決める: モデル A は除いたクリップをモデル B は含めたまま平均することになり、
    モデルごとに違うクリップ集合の CER を並べて比べることになるため。
    参照が空のクリップ（rate が None）は CER が定義できないので外れ値にしない
    """
    outliers: set[str] = set()
    for rows in rows_by_model.values():
        for clip_id, count in _norm_counts(rows):
            is_outlier = count.rate is not None and count.rate > OUTLIER_CER_NORM
            if is_outlier:
                outliers.add(clip_id)
    return sorted(outliers)


def outlier_summary(
    norm_counts: Sequence[tuple[str, EditCount]], outlier_ids: Collection[str]
) -> dict:
    """クリップ別 cer_norm の中央値・除いた外れ値の件数・run 共通の外れ値（outlier_ids）を除いたマイクロ平均。

    見出しの CER はすべてのクリップを含むマイクロ平均のままにし、これは感度分析として別に出す。
    Why not 外れ値を見出しから除く: 何を外れ値とみなすかで結果を動かせてしまい、モデル間の比較の公平さを損なうため。
    参照が空のクリップ（rate が None）は外れ値に数えず、除外後の平均にもそのまま残す（見出しと同じ扱い）
    """
    rates = [count.rate for _, count in norm_counts if count.rate is not None]
    excluded = [count for clip_id, count in norm_counts if clip_id in outlier_ids]
    kept = [count for clip_id, count in norm_counts if clip_id not in outlier_ids]
    return {
        "median": statistics.median(rates) if rates else None,
        "outliers": len(excluded),
        "excluding_outliers": _edit_count_json(_sum_counts(kept)),
    }


def summarize_clips(rows: Sequence[dict], outlier_ids: Collection[str]) -> dict:
    """1 グループ（モデル × データセット、または全体）の CER・速度・失敗数・外れ値の影響。

    outlier_ids は run 全体で決めた外れ値（run_outlier_ids）。どのモデル・グループでも同じ集合を除く。
    CER と RTF は失敗クリップを除いて計算し、失敗数は別に必ず出す。
    Why not 失敗を空の仮説として CER に含める: タイムアウトや GPU のエラーは認識精度とは別の問題で、
    混ぜると「よく聞き取れなかった」のか「動かなかった」のかが区別できなくなるため。
    """
    succeeded = [row for row in rows if not _is_failed(row)]
    summary: dict = {
        "clips": len(rows),
        "succeeded": len(succeeded),
        "failed": len(rows) - len(succeeded),
    }
    for metric in METRIC_NAMES:
        counts = [edit_count(row["reference"], row["hypothesis"], metric) for row in succeeded]
        summary[metric] = _edit_count_json(_sum_counts(counts))
    summary["cer_norm_clips"] = outlier_summary(_norm_counts(succeeded), outlier_ids)
    timing = speed([(row["duration_s"], row["decode_ms"], row["infer_ms"]) for row in succeeded])
    summary.update(
        {
            "rtf": timing.rtf,
            "rtf_with_decode": timing.rtf_with_decode,
            "audio_seconds": timing.audio_seconds,
        }
    )
    return summary


def worst_examples(rows: Iterable[dict], count: int = WORST_EXAMPLES) -> list[dict]:
    """cer_norm の高い順に count 件。ID と数値だけを返す（テキストは返さない）。"""
    scored = []
    for row in rows:
        if _is_failed(row):
            continue
        norm = edit_count(row["reference"], row["hypothesis"], "cer_norm")
        reading = edit_count(row["reference"], row["hypothesis"], "cer_reading")
        has_reference = norm.rate is not None
        if not has_reference:
            continue
        scored.append(
            {
                "id": row["id"],
                "cer_norm": norm.rate,
                "cer_reading": reading.rate,
                "edits": norm.edits,
                "reference_chars": norm.reference_chars,
            }
        )
    scored.sort(key=lambda example: (-example["cer_norm"], -example["edits"], example["id"]))
    return scored[:count]


def dataset_order(names: Iterable[str]) -> list[str]:
    """prepare の定義順（jsut, reazon, cv）を先に、それ以外は名前順。"""
    present = set(names)
    known = [name for name in DATASETS if name in present]
    return known + sorted(present - set(known))


def _first_value(rows: Iterable[dict], key: str) -> float | None:
    """最初に値が入っている行の値。セッションを開き直すと値が変わるので、最初のセッションの値を代表にする。"""
    for row in rows:
        value = row.get(key)
        if value is not None:
            return value
    return None


def summarize(rows_by_model: dict[str, list[dict]], metadata: dict) -> dict:
    model_metadata = metadata.get("models") or {}
    # run.json にあって raw が無いモデル（読み込みに失敗したなど）も「未計測」として残す
    ordered_models = list(model_metadata)
    ordered_models += sorted(set(rows_by_model) - set(ordered_models))
    outlier_ids = run_outlier_ids(rows_by_model)
    outlier_set = frozenset(outlier_ids)

    models: dict[str, dict] = {}
    for model in ordered_models:
        rows = rows_by_model.get(model, [])
        datasets = dataset_order(row["dataset"] for row in rows)
        info = model_metadata.get(model) or {}
        planned = info.get("planned")
        models[model] = {
            # 最初の計測セッションの読み込み時間。run_browser が先にダウンロード専用のセッションを開くので、
            # キャッシュからの読み込みになる（それ以前の run ではダウンロードを含みうる）
            "load_ms": _first_value(rows, "load_ms"),
            # ダウンロード専用セッションの読み込み時間。first_load_downloaded が True のときだけダウンロードを含む
            "first_load_ms": info.get("first_load_ms"),
            # そのセッションでモデルのファイルをダウンロードしたか（False はキャッシュから、None は不明。古い run も None）
            "first_load_downloaded": info.get("first_load_downloaded"),
            "warmup_ms": _first_value(rows, "warmup_ms"),
            "info": info.get("info"),
            "planned": planned,
            "missing": max(planned - len(rows), 0) if planned is not None else None,
            "status": info.get("status"),
            "datasets": {
                dataset: summarize_clips(
                    [row for row in rows if row["dataset"] == dataset], outlier_set
                )
                for dataset in datasets
            },
            OVERALL: summarize_clips(rows, outlier_set),
            "worst": {
                dataset: worst_examples(row for row in rows if row["dataset"] == dataset)
                for dataset in datasets
            },
            "failed_ids": [
                {"id": row["id"], "dataset": row["dataset"], "error": row["error"]}
                for row in rows
                if _is_failed(row)
            ],
        }
    return {
        "run_id": metadata.get("run_id"),
        "conditions": metadata,
        # ID だけ（テキストは載せない）。感度分析で全モデルから除いたクリップ
        "outlier_ids": outlier_ids,
        "models": models,
    }


def _round(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(value, digits)


def _seconds(milliseconds: float | None) -> float | None:
    return None if milliseconds is None else round(milliseconds / 1000, 3)


def _page_group(group: dict) -> dict:
    clips = group["cer_norm_clips"]
    return {
        "n": group["clips"],
        "failed": group["failed"],
        **{metric: _round(group[metric]["rate"]) for metric in METRIC_NAMES},
        "rtf": _round(group["rtf"]),
        "rtf_with_decode": _round(group["rtf_with_decode"]),
        "audio_s": _round(group["audio_seconds"], 3),
        "cer_norm_median": _round(clips["median"]),
        "cer_norm_outliers": clips["outliers"],
        "cer_norm_excluding_outliers": _round(clips["excluding_outliers"]["rate"]),
    }


def _gpu_text(gpu: dict | None) -> str | None:
    if not gpu:
        return None
    parts = [gpu.get(key) for key in ("vendor", "architecture", "device", "description")]
    return " / ".join(str(part) for part in parts if part) or None


def _environment(conditions: dict) -> dict:
    chrome = conditions.get("chrome") or {}
    host = conditions.get("host") or {}
    gpu = chrome.get("gpu")
    # host が無い古い run.json では、ブラウザが申告した OS で代用する
    browser_platform = " ".join(
        str(part) for part in (chrome.get("platform"), chrome.get("platform_version")) if part
    )
    return {
        "chrome": chrome.get("version"),
        "platform": host.get("platform") or browser_platform or None,
        "gpu": _gpu_text(gpu),
        "gpu_adapter": (
            {key: gpu.get(key) for key in ("vendor", "architecture", "device", "description")}
            if gpu
            else None
        ),
        "machine": host.get("machine"),
    }


def _page_datasets(summary: dict, conditions: dict) -> list[dict]:
    manifest = conditions.get("manifest") or {}
    created_from = manifest.get("created_from") or {}
    excluded = manifest.get("excluded_over_30s") or {}
    planned = (conditions.get("planned") or {}).get("datasets")
    if planned is None:
        # planned が無い古い run.json では、最初のモデルの行数と音声長で代用する
        first = next(iter(summary["models"].values()), None)
        groups = (first or {}).get("datasets", {})
        planned = {
            key: {"n": group["clips"], "audio_s": group["audio_seconds"]}
            for key, group in groups.items()
        }
    entries = []
    for key in dataset_order(planned):
        spec = DATASETS.get(key)
        source = created_from.get(key) or {}
        entries.append(
            {
                "key": key,
                "label": spec.label if spec else key,
                "repo": source.get("repo") or (spec.repo if spec else None),
                "revision": source.get("revision") or (spec.revision if spec else None),
                "n": planned[key]["n"],
                "excluded_over_30s": excluded.get(key),
                "total_audio_s": _round(planned[key]["audio_s"], 3),
            }
        )
    return entries


def _page_model(key: str, result: dict) -> dict:
    info = result.get("info") or {}
    has_rows = result[OVERALL]["clips"] > 0
    return {
        "key": key,
        "label": info.get("label") or key,
        "id": info.get("id"),
        "revision": info.get("revision"),
        "dtype": info.get("dtype"),
        "status": result["status"],
        "planned": result["planned"],
        "missing": result["missing"],
        "load_s": _seconds(result["load_ms"]),
        "first_load_s": _seconds(result["first_load_ms"]),
        # first_load_s がダウンロードを含むのは True のときだけ（False はキャッシュから、None は不明）
        "first_load_downloaded": result["first_load_downloaded"],
        "warmup_s": _seconds(result["warmup_ms"]),
        "per_dataset": {
            dataset: _page_group(group) for dataset, group in result["datasets"].items()
        },
        # 1 件も記録できなかったモデルは null（ページは「未計測」と表示する）
        "overall": _page_group(result[OVERALL]) if has_rows else None,
        "worst_ids": {
            dataset: [example["id"] for example in examples]
            for dataset, examples in result["worst"].items()
        },
        "failed_ids": [item["id"] for item in result["failed_ids"]],
    }


# ページは notes を「補足」としてそのまま表示する。指標の定義はページ側の「指標とデータ」が説明するので、
# ここにはそこに無い注意点だけを読者向けの言葉で書く。
# Why not JSON の項目名（cer_norm_outliers など）で書く: 一般の読者には意味が通じず、項目の意味はこのモジュールと README に書いてあるため
PAGE_NOTES = [
    "失敗したクリップ（タイムアウトや WebGPU のエラーなど）は CER と RTF から除き、失敗数として別に数えています",
    f"外れ値（どれか 1 つのモデルでクリップ単位の CER（正規化）が {OUTLIER_CER_NORM:.0%} を超えたクリップ）を除いた CER は感度を見るための参考値です。"
    "外れ値は全モデル共通で、どのモデルからも同じクリップを除いています。"
    "結論や最良値の判定は、すべてのクリップを含む CER で行っています",
    "ReazonSpeech には参照テキストが音声の一部しか書き起こしていないクリップがあり、正しく聞き取っても CER が 100% を超えることがあります",
    "ウォームアップには先頭のクリップを使い、その時間は RTF に含めていません",
    "初回の読み込み時間がダウンロードを含むのは、その読み込みでダウンロードが起きたと記録されているときだけです",
]


def build_page_json(summary: dict, generated_at: str) -> dict:
    """ベンチマークページ（benchmarks.html）が読む集計 JSON。数値・ID・実行環境だけで、テキストは含めない。"""
    conditions = summary["conditions"]
    git = conditions.get("git") or {}
    return {
        "schema": SCHEMA_VERSION,
        "run_id": summary["run_id"],
        "generated_at": generated_at,
        "git_commit": git.get("commit"),
        "git_dirty": git.get("dirty"),
        "started_at": conditions.get("started_at"),
        "finished_at": conditions.get("finished_at"),
        "language": conditions.get("language"),
        "environment": _environment(conditions),
        "outlier_cer_norm_threshold": OUTLIER_CER_NORM,
        # run 共通の外れ値（ID だけ）。各モデルの cer_norm_excluding_outliers はこの同じ集合を除いた値
        "outlier_ids": summary["outlier_ids"],
        "outlier_count": len(summary["outlier_ids"]),
        "datasets": _page_datasets(summary, conditions),
        "models": [_page_model(key, result) for key, result in summary["models"].items()],
        "notes": PAGE_NOTES,
    }


def _percent(value: float | None) -> str:
    return "–" if value is None else f"{value * 100:.1f}"


def _fixed(value: float | None, digits: int) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def _downloaded_text(downloaded: bool | None) -> str:
    if downloaded is None:
        return "不明"
    return "あり" if downloaded else "なし"


def _short(text: str, limit: int = 80) -> str:
    single_line = " ".join(str(text).split())
    is_long = len(single_line) > limit
    return single_line[: limit - 1] + "…" if is_long else single_line


def _conditions_lines(conditions: dict) -> list[str]:
    git = conditions.get("git") or {}
    commit = git.get("commit") or "不明"
    dirty = "（未コミットの変更あり）" if git.get("dirty") else ""
    environment = _environment(conditions)
    manifest = conditions.get("manifest") or {}
    excluded = manifest.get("excluded_over_30s") or {}
    excluded_text = ", ".join(f"{name} {count}" for name, count in excluded.items()) or "–"
    limit = conditions.get("limit")
    scope = f"offset {conditions.get('offset', 0)}" + (
        f", limit {limit}" if limit is not None else ""
    )
    resumes = len(conditions.get("resumes") or [])
    resumed = f"（途中から {resumes} 回再開）" if resumes else ""
    return [
        f"- 実行日時（UTC）: {conditions.get('started_at', '不明')} 〜 {conditions.get('finished_at', '不明')}{resumed}",
        f"- git commit: `{commit}`{dirty}",
        f"- Chrome: {environment['chrome'] or '不明'}",
        f"- GPU（WebGPU アダプタ）: {environment['gpu'] or '不明'}",
        f"- OS / 機種: {environment['platform'] or '不明'} / {environment['machine'] or '不明'}",
        f"- manifest: {manifest.get('items', '不明')} 件"
        f"（1 データセットあたり {manifest.get('per_dataset', '不明')} 件、"
        f"30 秒超で除外: {excluded_text}）、評価範囲: {scope}",
        f"- 言語: {conditions.get('language', '不明')}",
    ]


def _group_of(result: dict, group: str) -> dict | None:
    return result[OVERALL] if group == OVERALL else result["datasets"].get(group)


def _model_label(model: str, result: dict) -> str:
    label = (result.get("info") or {}).get("label")
    return f"{model}（{label}）" if label else model


def render_markdown(summary: dict) -> str:
    models = summary["models"]
    datasets = dataset_order(dataset for model in models.values() for dataset in model["datasets"])
    groups = [*datasets, OVERALL]
    short_names = {"cer_strict": "strict", "cer_norm": "norm", "cer_reading": "reading"}

    lines = [f"# 評価結果 {summary['run_id']}", "", "## 実行条件", ""]
    lines += _conditions_lines(summary["conditions"])
    lines += [
        "",
        "## 精度と速度",
        "",
        "CER は % 表示のマイクロ平均（総編集数 / 参照総文字数）。失敗したクリップは CER と RTF から除き、失敗数として別に示す。",
        "load 秒はキャッシュからの読み込み、初回 load 秒はダウンロード専用セッションでの最初の読み込み"
        "（ダウンロードを含むのは初回 DL が「あり」のときだけ）、"
        "warmup 秒は計測前に 1 回だけ流した文字起こし（計測には含めない）。",
        "",
    ]
    header = ["model"]
    for group in groups:
        label = "全体" if group == OVERALL else group
        header += [f"{label} {short_names[metric]}" for metric in METRIC_NAMES]
    header += ["RTF", "load 秒", "初回 load 秒", "初回 DL", "warmup 秒", "失敗"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|")
    for model, result in models.items():
        cells = [model]
        for group in groups:
            group_summary = _group_of(result, group)
            for metric in METRIC_NAMES:
                rate = group_summary[metric]["rate"] if group_summary else None
                cells.append(_percent(rate))
        overall = result[OVERALL]
        cells += [
            _fixed(overall["rtf"], 3),
            _fixed(_seconds(result["load_ms"]), 1),
            _fixed(_seconds(result["first_load_ms"]), 1),
            _downloaded_text(result["first_load_downloaded"]),
            _fixed(_seconds(result["warmup_ms"]), 1),
            str(overall["failed"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    model_infos = [
        (model, result["info"]) for model, result in models.items() if result.get("info")
    ]
    if model_infos:
        lines += ["", "| model | 名前 | リポジトリ | revision |", "|---|---|---|---|"]
        for model, info in model_infos:
            lines.append(
                f"| {model} | {info.get('label', '')} | {info.get('id', '')} "
                f"| `{info.get('revision', '')}` |"
            )

    lines += [
        "",
        "## 件数と速度の内訳",
        "",
        "| model | dataset | 件数 | 成功 | 失敗 | 音声 秒 | RTF | RTF（デコード込み） |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, result in models.items():
        for group in groups:
            group_summary = _group_of(result, group)
            if group_summary is None:
                continue
            label = "全体" if group == OVERALL else group
            lines.append(
                f"| {model} | {label} | {group_summary['clips']} | {group_summary['succeeded']} "
                f"| {group_summary['failed']} | {group_summary['audio_seconds']:.1f} "
                f"| {_fixed(group_summary['rtf'], 3)} | {_fixed(group_summary['rtf_with_decode'], 3)} |"
            )
    incomplete = {
        model: result
        for model, result in models.items()
        if result["missing"] or (result["status"] not in (None, "done"))
    }
    if incomplete:
        lines += ["", "未完了のモデル（予定件数に届いていない）:", ""]
        for model, result in incomplete.items():
            lines.append(f"- {model}: 未実行 {result['missing']} 件、終了状態 `{result['status']}`")

    outlier_ids = summary["outlier_ids"]
    lines += [
        "",
        "## 外れ値の影響（感度分析）",
        "",
        f"この run のどれか 1 つのモデルでクリップ単体の cer_norm が {OUTLIER_CER_NORM:.0%} を超えたクリップを外れ値とし、"
        "同じクリップをすべてのモデルから除く（モデルごとに違うクリップ集合で比べないため）。"
        "ReazonSpeech は参照テキストが音声の一部しか書き起こしていないクリップがあり、正しく聞き取っても 200〜300% になることがある。"
        "見出しの CER（上の表）は外れ値も含めた全件のマイクロ平均のままで、右端の列は外れ値を除いた場合の参考値。",
        "",
        f"外れ値: {len(outlier_ids)} 件（{', '.join(outlier_ids) or 'なし'}）",
        "",
        "| model | dataset | norm（全件） | クリップ別 norm の中央値 | 外れ値の件数 | norm（外れ値を除く・参考） |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model, result in models.items():
        for group in groups:
            group_summary = _group_of(result, group)
            if group_summary is None:
                continue
            clips = group_summary["cer_norm_clips"]
            label = "全体" if group == OVERALL else group
            lines.append(
                f"| {model} | {label} | {_percent(group_summary['cer_norm']['rate'])} "
                f"| {_percent(clips['median'])} | {clips['outliers']} "
                f"| {_percent(clips['excluding_outliers']['rate'])} |"
            )

    lines += [
        "",
        "## 誤りの大きい例",
        "",
        f"cer_norm の高い順に各 {WORST_EXAMPLES} 件。音声と参照テキストは再配布しないため ID だけを示す"
        "（`eval/data/manifest.json` と raw 結果で中身を確認できる）。",
        "",
        "| model | dataset | id | cer_norm | cer_reading | 編集数 | 参照文字数 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for model, result in models.items():
        for dataset, examples in result["worst"].items():
            for example in examples:
                lines.append(
                    f"| {model} | {dataset} | {example['id']} | {_percent(example['cer_norm'])} "
                    f"| {_percent(example['cer_reading'])} | {example['edits']} "
                    f"| {example['reference_chars']} |"
                )

    failed = [(model, item) for model, result in models.items() for item in result["failed_ids"]]
    if failed:
        lines += ["", "## 失敗したクリップ", "", "| model | id | エラー |", "|---|---|---|"]
        for model, item in failed:
            lines.append(
                f"| {model} | {item['id']} | {_short(item['error']).replace('|', '\\|')} |"
            )

    lines += [
        "",
        "## 指標",
        "",
        "- strict: NFC と空白の除去だけ。全角半角・句読点・漢字/かなの違いもすべて誤りに数える",
        "- norm: NFKC、句読点と記号の除去、英字の小文字化。一般的な日本語 ASR 評価に近い",
        "- reading: norm の後に形態素解析（fugashi + unidic-lite）で読み（カタカナ）にしてから比べる。"
        "漢字の選び方や送り仮名の違いを無視した「音の正しさ」",
        "- RTF: 推論時間の合計 / 音声長の合計。1 未満なら実時間より速い。ウォームアップの 1 回は含めない",
        "- load 秒: キャッシュ済みのモデルの読み込み時間。初回 load 秒は初回 DL が「あり」のときだけダウンロードを含む",
        "",
    ]
    return "\n".join(lines)


def latest_run_id(raw_dir: Path) -> str | None:
    """run_id は UTC のタイムスタンプなので、名前順の最後が最新。形式に合わない名前のディレクトリは数えない。"""
    if not raw_dir.is_dir():
        return None
    runs = run_ids(path.name for path in raw_dir.iterdir() if path.is_dir())
    return runs[-1] if runs else None


def is_newest_scored(out_dir: Path, run_id: str) -> bool:
    """out_dir にある集計 JSON の中で run_id が最新（名前順で最後）か。

    古い run を集計し直したときに latest.json を巻き戻さないため。
    run_id の形式に合わない JSON（latest.json や手で置いたファイル）は比べる相手にしない。
    """
    scored = run_ids(path.stem for path in out_dir.glob("*.json"))
    return all(run_id >= other for other in scored)


def write_text(path: Path, text: str) -> None:
    # 途中で落ちても壊れた JSON をページに読ませないよう、一時ファイルに書いてから置き換える
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="", help="空なら最新の run")
    parser.add_argument("--raw-dir", type=Path, default=RAW_RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    try:
        given = reassign_named({"run": args.run_id})
    except ValueError as error:
        log("invalid_arguments", message=str(error))
        return 2
    requested = given["run"].strip()
    is_invalid_run_id = requested != "" and not is_valid_run_id(requested)
    if is_invalid_run_id:
        log("invalid_arguments", message=f"run_id は YYYYMMDDTHHMMSSZ の形式です: {requested}")
        return 2
    run_id = requested or latest_run_id(args.raw_dir)
    if run_id is None:
        log("no_runs", raw_dir=str(args.raw_dir), hint="just eval-run")
        return 1
    run_dir = args.raw_dir / run_id
    rows_by_model = load_rows(run_dir)
    if not rows_by_model:
        log("no_results", run_dir=str(run_dir))
        return 1

    metadata_path = run_dir / "run.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    )
    metadata.setdefault("run_id", run_id)

    summary = summarize(rows_by_model, metadata)
    summary_path = run_dir / "summary.json"
    write_text(summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = args.out_dir / f"{run_id}.md"
    write_text(markdown_path, render_markdown(summary))
    generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    page_text = (
        json.dumps(build_page_json(summary, generated_at), ensure_ascii=False, indent=2) + "\n"
    )
    page_path = args.out_dir / f"{run_id}.json"
    write_text(page_path, page_text)
    is_latest = is_newest_scored(args.out_dir, run_id)
    if is_latest:
        write_text(args.out_dir / f"{LATEST_NAME}.json", page_text)
    log(
        "scored",
        run_id=run_id,
        markdown=str(markdown_path),
        page_json=str(page_path),
        latest_updated=is_latest,
        summary=str(summary_path),
        models={
            model: {
                "cer_norm": result[OVERALL]["cer_norm"]["rate"],
                "rtf": result[OVERALL]["rtf"],
                "failed": result[OVERALL]["failed"],
            }
            for model, result in summary["models"].items()
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
