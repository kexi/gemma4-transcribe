"""prepare.py が保証すること: サンプリングの決定性・30 秒超の除外・manifest の並び・音声の形式（ネットワーク不要）。"""

import hashlib
import io

import numpy as np
import pytest
import soundfile

from gemma4_eval.prepare import (
    DATASETS,
    TARGET_SAMPLE_RATE,
    decode_to_mono_16k,
    interleave,
    item_id,
    sampling_order,
    take_within_duration,
)


class TestSamplingOrder:
    def test_is_sorted_by_sha256_of_dataset_and_row(self):
        order = sampling_order("jsut", 50)
        keys = [hashlib.sha256(f"jsut:{row}".encode()).hexdigest() for row in order]
        assert keys == sorted(keys)
        assert sorted(order) == list(range(50))

    def test_is_identical_across_calls(self):
        assert sampling_order("cv", 1000) == sampling_order("cv", 1000)

    def test_known_prefix_is_stable(self):
        """Python の版や hash seed が変わっても同じ行が選ばれる（過去の結果と比較できる）。"""
        expected = sorted(
            range(5000), key=lambda row: hashlib.sha256(f"jsut:{row}".encode()).digest()
        )
        assert sampling_order("jsut", 5000)[:5] == expected[:5]

    def test_differs_between_datasets(self):
        assert sampling_order("jsut", 100) != sampling_order("reazon", 100)

    def test_smaller_sample_is_prefix_of_larger_sample(self):
        """N を増やしても既に選ばれた行は変わらない（per-dataset 3 の結果は 100 の部分集合）。"""
        durations = {row: 40.0 if row % 3 == 0 else 5.0 for row in range(200)}

        def select(count):
            kept, _ = take_within_duration(
                sampling_order("reazon", 200), count, lambda row: row, durations.__getitem__
            )
            return kept

        small, large = select(3), select(50)
        assert len(large) == 50
        assert large[:3] == small


class TestTakeWithinDuration:
    def test_skips_clips_over_limit_and_counts_them(self):
        durations = {0: 10.0, 1: 31.0, 2: 30.0, 3: 45.0, 4: 1.0, 5: 2.0}
        kept, excluded = take_within_duration(
            [0, 1, 2, 3, 4, 5], 3, lambda row: (row, durations[row]), lambda clip: clip[1]
        )
        assert [row for row, _ in kept] == [0, 2, 4]
        assert excluded == 2

    def test_exactly_thirty_seconds_is_kept(self):
        kept, excluded = take_within_duration([0], 1, lambda row: 30.0, float)
        assert kept == [30.0]
        assert excluded == 0

    def test_stops_loading_once_enough_clips_are_kept(self):
        """必要数が集まったら残りの候補はデコードしない。"""
        loaded = []

        def load(row):
            loaded.append(row)
            return row

        take_within_duration(range(100), 2, load, lambda _: 1.0)
        assert loaded == [0, 1]

    def test_returns_fewer_when_candidates_run_out(self):
        kept, excluded = take_within_duration([0, 1], 5, lambda row: 40.0 if row else 1.0, float)
        assert kept == [1.0]
        assert excluded == 1

    def test_same_candidates_give_same_selection(self):
        """同じ候補順と長さなら、何度実行しても同じ行が選ばれる。"""
        durations = {row: (row * 7919) % 50 for row in range(300)}

        def select():
            kept, excluded = take_within_duration(
                sampling_order("cv", 300), 20, lambda row: row, durations.__getitem__
            )
            return kept, excluded

        assert select() == select()


class TestManifestOrder:
    def test_interleave_alternates_datasets(self):
        """limit で先頭だけ評価しても全データセットが含まれるよう、交互に並べる。"""
        assert interleave([["j1", "j2", "j3"], ["r1"], ["c1", "c2"]]) == [
            "j1",
            "r1",
            "c1",
            "j2",
            "c2",
            "j3",
        ]

    def test_interleave_of_nothing_is_empty(self):
        assert interleave([]) == []

    def test_item_id_is_zero_padded_row(self):
        assert item_id("jsut", 123) == "jsut-000123"


class TestDatasets:
    def test_revisions_are_pinned_commit_shas(self):
        for spec in DATASETS.values():
            assert len(spec.revision) == 40
            int(spec.revision, 16)


class TestDecodeToMono16k:
    @pytest.mark.parametrize("sample_rate", [16_000, 44_100, 48_000])
    def test_downmixes_and_resamples_to_16k(self, sample_rate):
        seconds = 0.5
        frames = int(sample_rate * seconds)
        time = np.arange(frames) / sample_rate
        stereo = np.stack([np.sin(2 * np.pi * 440 * time), np.zeros(frames)], axis=1) * 0.5
        buffer = io.BytesIO()
        soundfile.write(buffer, stereo, sample_rate, format="WAV", subtype="PCM_16")

        samples = decode_to_mono_16k(buffer.getvalue())

        assert samples.ndim == 1
        assert samples.dtype == np.float32
        assert len(samples) == pytest.approx(TARGET_SAMPLE_RATE * seconds, abs=2)
        # 片チャンネル無音のステレオを平均するので振幅は半分になる
        assert np.max(np.abs(samples)) == pytest.approx(0.25, abs=0.02)
