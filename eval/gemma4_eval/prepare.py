"""評価データの準備: 固定 revision の parquet を取得し、決定的に N 件を選んで 16kHz mono PCM16 wav と manifest を書く。

使い方: uv run --project eval python -m gemma4_eval.prepare --per-dataset 100
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .just_args import reassign_named
from .log import log
from .paths import DATA_DIR

TARGET_SAMPLE_RATE = 16_000
# Gemma 4 の音声入力 1 区間の上限。超える音声はアプリでは分割されるが、
# 分割境界の影響を混ぜずにモデル単体の精度を測るため評価対象から外す
MAX_DURATION_S = 30.0
DEFAULT_PER_DATASET = 100


@dataclass(frozen=True)
class DatasetSpec:
    repo: str
    revision: str
    split: str
    # 集計 JSON・ベンチマークページに出す表示名
    label: str


# revision はコミット SHA で固定する。main を追うとデータの差し替えで過去の結果と比較できなくなるため
DATASETS: dict[str, DatasetSpec] = {
    "jsut": DatasetSpec(
        "japanese-asr/ja_asr.jsut_basic5000",
        "278db379fc96167ff2293d7abf9ab86976afcd78",
        "test",
        "JSUT basic5000",
    ),
    "reazon": DatasetSpec(
        "japanese-asr/ja_asr.reazonspeech_test",
        "dd08bfb9dfc1cef4e4d0609fd78c3755d48b926f",
        "test",
        "ReazonSpeech test",
    ),
    "cv": DatasetSpec(
        "japanese-asr/ja_asr.common_voice_8_0",
        "bf8819e8d9a5feb51b0c718686bd20ea67a3c729",
        "test",
        "Common Voice 8.0（ja）",
    ),
}


def sampling_key(dataset: str, row: int) -> str:
    """行 row の並び順を決めるキー。sha256 なので Python の版や hash seed に依存しない。"""
    return hashlib.sha256(f"{dataset}:{row}".encode()).hexdigest()


def sampling_order(dataset: str, num_rows: int) -> list[int]:
    """全行を sampling_key の昇順に並べた行番号。先頭から取れば N を増やしても既存の選択は変わらない。"""
    return sorted(range(num_rows), key=lambda row: sampling_key(dataset, row))


def take_within_duration[T](
    candidates: Iterable[int],
    count: int,
    load: Callable[[int], T],
    duration_of: Callable[[T], float],
    max_duration_s: float = MAX_DURATION_S,
) -> tuple[list[T], int]:
    """候補を順に読み、max_duration_s 以下のものを count 件集める。

    戻り値の 2 つ目は、集め終わるまでに長すぎて飛ばした件数。
    Why not 全行の長さを先に測る: 全行（数千件）の音声をデコードする必要があり、選ばれる行は同じなので無駄になる。
    """
    kept: list[T] = []
    excluded = 0
    for row in candidates:
        is_enough = len(kept) >= count
        if is_enough:
            break
        clip = load(row)
        is_too_long = duration_of(clip) > max_duration_s
        if is_too_long:
            excluded += 1
            continue
        kept.append(clip)
    return kept, excluded


def interleave[T](groups: Sequence[Sequence[T]]) -> list[T]:
    """各グループから 1 件ずつ順に取り出して並べる。

    manifest をデータセット順ではなく交互に並べるのは、評価ページの limit で先頭だけ流したときにも
    全データセットが含まれるようにするため。
    """
    result: list[T] = []
    longest = max((len(group) for group in groups), default=0)
    for index in range(longest):
        for group in groups:
            has_item = index < len(group)
            if has_item:
                result.append(group[index])
    return result


def item_id(dataset: str, row: int) -> str:
    return f"{dataset}-{row:06d}"


@dataclass
class Clip:
    dataset: str
    row: int
    reference: str
    samples: (
        object  # numpy.ndarray（float32, 16kHz mono）。numpy を型注釈のためだけに import しない
    )
    duration_s: float


def decode_to_mono_16k(audio_bytes: bytes):
    """wav / flac / mp3 のバイト列を 16kHz mono float32 にする（Web Audio の decodeToMono16k と同じ形）。"""
    import numpy as np
    import soundfile
    import soxr

    data, sample_rate = soundfile.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    needs_resample = sample_rate != TARGET_SAMPLE_RATE
    if needs_resample:
        mono = soxr.resample(mono, sample_rate, TARGET_SAMPLE_RATE, quality="HQ")
    return np.clip(mono, -1.0, 1.0).astype(np.float32)


class ShardedParquet:
    """複数 shard の parquet を、shard を並べた通し行番号で引けるようにする。

    行は sha256 順に飛び飛びで読むので、row group（約 100 行）単位で直近のものだけを保持する。
    Why not datasets ライブラリ: Audio 特徴量のデコードに torchcodec 等の重い依存が入り、
    データセット全体を Arrow キャッシュに展開するため、必要な数十件のためには大きすぎる。
    """

    def __init__(self, paths: Sequence[Path]) -> None:
        import pyarrow.parquet as pq

        self._files = [pq.ParquetFile(path) for path in paths]
        # (shard, row_group, 先頭の通し行番号, 行数)
        self._groups: list[tuple[int, int, int, int]] = []
        start = 0
        for shard, parquet_file in enumerate(self._files):
            metadata = parquet_file.metadata
            for group in range(metadata.num_row_groups):
                rows = metadata.row_group(group).num_rows
                self._groups.append((shard, group, start, rows))
                start += rows
        self.num_rows = start
        self._cache: dict[tuple[int, int], list[dict]] = {}

    def row(self, index: int) -> dict:
        is_out_of_range = not 0 <= index < self.num_rows
        if is_out_of_range:
            raise IndexError(index)
        for shard, group, start, rows in self._groups:
            is_inside = start <= index < start + rows
            if not is_inside:
                continue
            key = (shard, group)
            if key not in self._cache:
                self._cache.clear()
                table = self._files[shard].read_row_group(group, columns=["audio", "transcription"])
                self._cache[key] = table.to_pylist()
            return self._cache[key][index - start]
        raise IndexError(index)


def download_shards(spec: DatasetSpec) -> list[Path]:
    """固定 revision の parquet shard を HF キャッシュに取得し、shard 名順のローカルパスを返す。"""
    from huggingface_hub import HfApi, hf_hub_download

    # 公開データセットなので匿名で取得する。
    # Why not 既定のトークン: 失効したトークンがキャッシュに残っていると公開リポジトリでも 401 になるため
    api = HfApi(token=False)
    prefix = f"data/{spec.split}-"
    files = sorted(
        name
        for name in api.list_repo_files(spec.repo, repo_type="dataset", revision=spec.revision)
        if name.startswith(prefix) and name.endswith(".parquet")
    )
    if not files:
        raise RuntimeError(f"{spec.repo}@{spec.revision} に {prefix}*.parquet がありません")
    paths = []
    for filename in files:
        log("download", repo=spec.repo, revision=spec.revision, file=filename)
        local = hf_hub_download(
            spec.repo,
            filename,
            repo_type="dataset",
            revision=spec.revision,
            token=False,
        )
        paths.append(Path(local))
    return paths


def _load_clip(source: ShardedParquet, dataset: str, row: int) -> Clip:
    record = source.row(row)
    samples = decode_to_mono_16k(record["audio"]["bytes"])
    return Clip(
        dataset=dataset,
        row=row,
        reference=record["transcription"],
        samples=samples,
        duration_s=len(samples) / TARGET_SAMPLE_RATE,
    )


def _write_wav(path: Path, samples) -> None:
    import soundfile

    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(path, samples, TARGET_SAMPLE_RATE, subtype="PCM_16")


def _remove_stale_wavs(directory: Path, keep: set[str]) -> int:
    """前回の実行で選ばれ、今回は選ばれなかった wav を消す（manifest に無い音声を残さない）。"""
    removed = 0
    if not directory.is_dir():
        return removed
    for path in directory.glob("*.wav"):
        is_stale = path.name not in keep
        if is_stale:
            path.unlink()
            removed += 1
    return removed


def prepare(datasets: Sequence[str], per_dataset: int, out_dir: Path) -> dict:
    """データセットごとに per_dataset 件を選んで wav を書き、manifest（dict）を返す。"""
    created_from: dict[str, dict] = {}
    excluded_over_30s: dict[str, int] = {}
    groups: list[list[dict]] = []

    for dataset in datasets:
        spec = DATASETS[dataset]
        source = ShardedParquet(download_shards(spec))
        created_from[dataset] = {
            "repo": spec.repo,
            "revision": spec.revision,
            "split": spec.split,
            "num_rows": source.num_rows,
        }
        clips, excluded = take_within_duration(
            sampling_order(dataset, source.num_rows),
            per_dataset,
            load=lambda row, source=source, dataset=dataset: _load_clip(source, dataset, row),
            duration_of=lambda clip: clip.duration_s,
        )
        excluded_over_30s[dataset] = excluded

        items = []
        for clip in clips:
            clip_id = item_id(dataset, clip.row)
            relative = f"{dataset}/{clip_id}.wav"
            _write_wav(out_dir / relative, clip.samples)
            items.append(
                {
                    "id": clip_id,
                    "dataset": dataset,
                    "path": relative,
                    "duration_s": round(clip.duration_s, 3),
                    "reference": clip.reference,
                }
            )
        removed = _remove_stale_wavs(out_dir / dataset, {f"{item['id']}.wav" for item in items})
        log(
            "dataset_prepared",
            dataset=dataset,
            num_rows=source.num_rows,
            selected=len(items),
            excluded_over_30s=excluded,
            removed_stale=removed,
        )
        groups.append(items)

    return {
        "created_from": created_from,
        "per_dataset": per_dataset,
        "excluded_over_30s": excluded_over_30s,
        "items": interleave(groups),
    }


def write_manifest(manifest: dict, path: Path) -> None:
    # 途中で落ちても壊れた manifest を残さないよう、一時ファイルに書いてから置き換える
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # 文字列で受けるのは、just の `per=3` が "per=3" のまま渡ってくるため（reassign_named で振り直す）
    parser.add_argument("--per-dataset", default=str(DEFAULT_PER_DATASET))
    parser.add_argument(
        "--datasets",
        default=",".join(DATASETS),
        help=f"カンマ区切り（既定: {','.join(DATASETS)}）",
    )
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args(argv)

    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    unknown = [name for name in datasets if name not in DATASETS]
    if unknown:
        parser.error(f"未知のデータセット: {', '.join(unknown)}")
    try:
        per_dataset = int(reassign_named({"per": args.per_dataset})["per"])
    except ValueError as error:
        parser.error(f"--per-dataset は整数: {error}")
    is_invalid_count = per_dataset < 1
    if is_invalid_count:
        parser.error("--per-dataset は 1 以上")

    manifest = prepare(datasets, per_dataset, args.out_dir)
    manifest_path = args.out_dir / "manifest.json"
    write_manifest(manifest, manifest_path)
    log(
        "manifest_written",
        path=str(manifest_path),
        items=len(manifest["items"]),
        excluded_over_30s=manifest["excluded_over_30s"],
    )


if __name__ == "__main__":
    main()
