"""score.py が保証すること: 失敗クリップを CER から除きつつ必ず数えること、外れ値を見出しから除かないこと、
感度分析の外れ値を run 共通のクリップ集合で全モデルから除くこと、
Markdown と集計 JSON にテキストを載せないこと、集計 JSON がベンチマークページとの契約の形であること。"""

import json

import pytest

from gemma4_eval.metrics import EditCount
from gemma4_eval.paths import REPO_ROOT
from gemma4_eval.score import (
    OVERALL,
    SCHEMA_VERSION,
    build_page_json,
    is_newest_scored,
    latest_run_id,
    load_rows,
    main,
    outlier_summary,
    render_markdown,
    run_outlier_ids,
    summarize,
    summarize_clips,
    worst_examples,
)


def make_row(clip_id, dataset, reference, hypothesis, error=None, duration_s=2.0, infer_ms=1000.0):
    return {
        "id": clip_id,
        "dataset": dataset,
        "reference": reference,
        "hypothesis": hypothesis,
        "duration_s": duration_s,
        "decode_ms": 10.0,
        "infer_ms": infer_ms,
        "segments": 1,
        "error": error,
        "model": "e2b",
        "run_id": "20261004T000000Z",
        "load_ms": 5000.0,
        "warmup_ms": 800.0,
    }


ROWS = [
    make_row("jsut-000001", "jsut", "取り扱い説明書です。", "取扱い説明書です"),
    make_row("jsut-000002", "jsut", "今日は晴れ", "今日は晴れ"),
    make_row("cv-000003", "cv", "秘密の参照文", "", error="timeout: 300 秒", infer_ms=300_000.0),
]


class TestSummarizeClips:
    def test_failed_clips_are_counted_but_excluded_from_cer_and_rtf(self):
        summary = summarize_clips(ROWS, frozenset())
        assert summary["clips"] == 3
        assert summary["failed"] == 1
        # 成功 2 件だけ: 参照 norm 文字数 9 + 5、編集 1
        assert summary["cer_norm"]["edits"] == 1
        assert summary["cer_norm"]["reference_chars"] == 14
        assert summary["cer_reading"]["edits"] == 0
        assert summary["rtf"] == pytest.approx(2000 / 4 / 1000)

    def test_all_failed_group_has_undefined_cer(self):
        summary = summarize_clips([ROWS[2]], frozenset())
        assert summary["cer_norm"]["rate"] is None
        assert summary["rtf"] is None


class TestOutliers:
    def test_median_outlier_count_and_micro_average_without_outliers(self):
        """クリップ別 cer_norm: 0.1, 0.5, 3.0（外れ値 c）→ 中央値 0.5、外れ値 1 件、除外後は (1 + 5) / (10 + 10)。"""
        counts = [("a", EditCount(1, 10)), ("b", EditCount(5, 10)), ("c", EditCount(12, 4))]
        result = outlier_summary(counts, {"c"})
        assert result["median"] == pytest.approx(0.5)
        assert result["outliers"] == 1
        assert result["excluding_outliers"]["rate"] == pytest.approx(6 / 20)

    def test_excludes_the_given_ids_even_when_this_model_got_them_right(self):
        """別のモデルで外れ値になったクリップは、このモデルの CER が低くても除く（全モデルで同じクリップ集合にする）。"""
        counts = [("a", EditCount(1, 10)), ("b", EditCount(0, 10))]
        result = outlier_summary(counts, {"b"})
        assert result["outliers"] == 1
        assert result["excluding_outliers"]["rate"] == pytest.approx(1 / 10)
        # 中央値は除外前の全クリップで計算する
        assert result["median"] == pytest.approx(0.05)

    def test_exactly_one_hundred_percent_is_not_an_outlier(self):
        rows = [make_row("jsut-000001", "jsut", "あいう", "かきく")]
        assert run_outlier_ids({"e2b": rows}) == []

    def test_empty_reference_is_neither_outlier_nor_dropped(self):
        """参照が空のクリップは CER が定義できないので外れ値に数えず、見出しと同じく挿入数だけ分子に残す。"""
        rows = [
            make_row("cv-000001", "cv", "", "えーと"),
            make_row("cv-000002", "cv", "今日は晴れです", "今日は晴れです"),
        ]
        assert run_outlier_ids({"e2b": rows}) == []
        result = outlier_summary([("a", EditCount(3, 0)), ("b", EditCount(1, 10))], set())
        assert result["outliers"] == 0
        assert result["median"] == pytest.approx(0.1)
        assert result["excluding_outliers"]["edits"] == 4

    def test_run_outliers_are_the_union_over_models_and_ignore_failed_clips(self):
        """どれか 1 つのモデルで 100% を超えたクリップを、run 全体の外れ値とする（失敗クリップは判定に使わない）。"""
        rows_by_model = {
            "e2b": [
                make_row("reazon-000001", "reazon", "はい", "はいそうですねこれは"),
                make_row("reazon-000002", "reazon", "今日は晴れです", "今日は晴れです"),
            ],
            "e4b": [
                make_row("reazon-000001", "reazon", "はい", "はい"),
                make_row(
                    "reazon-000002", "reazon", "今日は晴れです", "あしたはあめがふるでしょうね"
                ),
                make_row("reazon-000003", "reazon", "秘密", "", error="timeout"),
            ],
        }
        assert run_outlier_ids(rows_by_model) == ["reazon-000001", "reazon-000002"]

    def test_every_model_is_compared_on_the_same_clips(self):
        """外れ値を除いた CER は、どのモデルでも同じクリップ集合（run 共通の外れ値を除いた残り）で計算する。"""
        rows_by_model = {
            "e2b": [
                make_row("reazon-000001", "reazon", "はい", "はいそうですねこれは"),
                make_row("reazon-000002", "reazon", "今日は晴れです", "今日は晴れです"),
            ],
            "e4b": [
                make_row("reazon-000001", "reazon", "はい", "はい"),
                make_row("reazon-000002", "reazon", "今日は晴れです", "今日は晴れでした"),
            ],
        }
        summary = summarize(rows_by_model, {"run_id": "20261004T000000Z"})
        assert summary["outlier_ids"] == ["reazon-000001"]
        for model in ("e2b", "e4b"):
            clips = summary["models"][model][OVERALL]["cer_norm_clips"]
            assert clips["outliers"] == 1
            # 残るのは reazon-000002 だけ（参照 7 文字）
            assert clips["excluding_outliers"]["reference_chars"] == 7

    def test_headline_cer_keeps_outlier_clips(self):
        """見出しの CER（マイクロ平均）は外れ値を除かず、除外後の値は別のキーに出す。"""
        rows = [
            make_row("reazon-000001", "reazon", "はい", "はいそうですねこれは"),
            make_row("reazon-000002", "reazon", "今日は晴れです", "今日は晴れです"),
        ]
        summary = summarize_clips(rows, run_outlier_ids({"e2b": rows}))
        assert summary["cer_norm"]["edits"] == 8
        assert summary["cer_norm"]["reference_chars"] == 9
        assert summary["cer_norm_clips"]["outliers"] == 1
        assert summary["cer_norm_clips"]["excluding_outliers"]["rate"] == 0


class TestWorstExamples:
    def test_orders_by_cer_norm_and_returns_ids_without_text(self):
        examples = worst_examples(ROWS)
        assert [example["id"] for example in examples] == ["jsut-000001", "jsut-000002"]
        assert set(examples[0]) == {"id", "cer_norm", "cer_reading", "edits", "reference_chars"}


class TestRender:
    def test_markdown_shows_failures_and_never_contains_reference_or_hypothesis_text(self):
        metadata = {
            "run_id": "20261004T000000Z",
            "models": {"e2b": {"planned": 4, "status": "done"}},
            "chrome": {"version": "140.0.1.2"},
        }
        summary = summarize({"e2b": ROWS}, metadata)
        markdown = render_markdown(summary)

        assert summary["models"]["e2b"]["missing"] == 1
        assert summary["models"]["e2b"][OVERALL]["failed"] == 1
        assert "cv-000003" in markdown
        assert "140.0.1.2" in markdown
        for row in ROWS:
            assert row["reference"] not in markdown
            is_nonempty_hypothesis = bool(row["hypothesis"])
            if is_nonempty_hypothesis:
                assert row["hypothesis"] not in markdown


class TestLoadRows:
    def test_skips_broken_last_line_and_duplicate_ids(self, tmp_path):
        """書き込み中に落ちた最終行は捨て、そこまでの結果は使う。"""
        lines = [json.dumps(ROWS[0]), json.dumps(ROWS[0]), json.dumps(ROWS[1]), '{"id": "jsu']
        (tmp_path / "e2b.jsonl").write_text("\n".join(lines), encoding="utf-8")
        rows = load_rows(tmp_path)
        assert [row["id"] for row in rows["e2b"]] == ["jsut-000001", "jsut-000002"]


METADATA = {
    "run_id": "20261004T000000Z",
    "started_at": "2026-10-04T00:00:00+00:00",
    "finished_at": "2026-10-04T00:10:00+00:00",
    "git": {"commit": "abc123", "dirty": True},
    "language": "Japanese",
    "offset": 0,
    "limit": None,
    "manifest": {
        "items": 3,
        "per_dataset": 2,
        "excluded_over_30s": {"jsut": 0, "cv": 1},
        "created_from": {"jsut": {"repo": "r/jsut", "revision": "1" * 40}},
    },
    "planned": {
        "items": 3,
        "datasets": {"jsut": {"n": 2, "audio_s": 4.0}, "cv": {"n": 1, "audio_s": 2.0}},
    },
    "models": {
        "e2b": {
            "planned": 3,
            "status": "done",
            "first_load_ms": 90_000.0,
            "first_load_downloaded": True,
            "info": {
                "key": "e2b",
                "label": "Gemma 4 E2B（q4f16）",
                "id": "onnx-community/gemma-4-E2B-it-ONNX",
                "revision": "9" * 40,
                "dtype": "q4f16",
            },
        },
        "e4b": {"planned": 3, "status": "prefetch-failed: load-error"},
    },
    "chrome": {
        "version": "154.0.1.2",
        "gpu": {"vendor": "apple", "architecture": "metal-3", "device": "", "description": ""},
    },
    "host": {"platform": "macOS 26.0 (arm64)", "machine": "Mac14,5 / Apple M2 Max"},
}


class TestPageJson:
    def page(self):
        return build_page_json(summarize({"e2b": ROWS}, METADATA), "2026-10-04T01:00:00+00:00")

    def test_has_the_benchmark_page_contract_shape(self):
        """benchmarks.html が依存するキー（bench-page-spec の schema 1）がすべてある。"""
        page = self.page()
        assert page["schema"] == SCHEMA_VERSION == 1
        assert page["run_id"] == "20261004T000000Z"
        assert page["generated_at"] == "2026-10-04T01:00:00+00:00"
        assert page["git_commit"] == "abc123"
        assert page["git_dirty"] is True
        assert page["environment"] == {
            "chrome": "154.0.1.2",
            "platform": "macOS 26.0 (arm64)",
            "gpu": "apple / metal-3",
            "gpu_adapter": {
                "vendor": "apple",
                "architecture": "metal-3",
                "device": "",
                "description": "",
            },
            "machine": "Mac14,5 / Apple M2 Max",
        }
        assert page["datasets"] == [
            {
                "key": "jsut",
                "label": "JSUT basic5000",
                "repo": "r/jsut",
                "revision": "1" * 40,
                "n": 2,
                "excluded_over_30s": 0,
                "total_audio_s": 4.0,
            },
            {
                "key": "cv",
                "label": "Common Voice 8.0（ja）",
                "repo": "japanese-asr/ja_asr.common_voice_8_0",
                "revision": "bf8819e8d9a5feb51b0c718686bd20ea67a3c729",
                "n": 1,
                "excluded_over_30s": 1,
                "total_audio_s": 2.0,
            },
        ]
        assert isinstance(page["notes"], list)
        assert page["notes"]

    def test_model_entry_has_load_warmup_cer_and_outlier_fields(self):
        model = self.page()["models"][0]
        assert model["key"] == "e2b"
        assert model["label"] == "Gemma 4 E2B（q4f16）"
        assert model["id"] == "onnx-community/gemma-4-E2B-it-ONNX"
        assert model["load_s"] == pytest.approx(5.0)
        assert model["first_load_s"] == pytest.approx(90.0)
        assert model["first_load_downloaded"] is True
        assert model["warmup_s"] == pytest.approx(0.8)
        expected_keys = {
            "n",
            "failed",
            "cer_strict",
            "cer_norm",
            "cer_reading",
            "rtf",
            "rtf_with_decode",
            "audio_s",
            "cer_norm_median",
            "cer_norm_outliers",
            "cer_norm_excluding_outliers",
        }
        assert set(model["overall"]) == expected_keys
        assert set(model["per_dataset"]) == {"jsut", "cv"}
        assert set(model["per_dataset"]["jsut"]) == expected_keys
        assert model["per_dataset"]["cv"]["failed"] == 1
        assert model["overall"]["n"] == 3
        assert model["worst_ids"]["jsut"] == ["jsut-000001", "jsut-000002"]
        assert model["failed_ids"] == ["cv-000003"]

    def test_model_without_results_is_listed_as_unmeasured(self):
        """読み込みに失敗して 1 件も無いモデルも載せ、overall を null にする（ページは「未計測」と表示する）。"""
        unmeasured = self.page()["models"][1]
        assert unmeasured["key"] == "e4b"
        assert unmeasured["overall"] is None
        assert unmeasured["per_dataset"] == {}
        assert unmeasured["status"] == "prefetch-failed: load-error"
        # run.json に記録が無い（古い run や読み込み失敗）なら不明として null
        assert unmeasured["first_load_downloaded"] is None

    def test_lists_run_level_outlier_ids_once_without_text(self):
        """外れ値はモデルごとではなく run に 1 回だけ、ID と件数で載せる。"""
        rows = [*ROWS, make_row("reazon-000001", "reazon", "はい", "はいそうですねこれは")]
        page = build_page_json(summarize({"e2b": rows}, METADATA), "2026-10-04T01:00:00+00:00")
        assert page["outlier_ids"] == ["reazon-000001"]
        assert page["outlier_count"] == 1
        assert "はいそうですね" not in json.dumps(page, ensure_ascii=False)

    def test_never_contains_reference_or_hypothesis_text(self):
        text = json.dumps(self.page(), ensure_ascii=False)
        for row in ROWS:
            assert row["reference"] not in text
            is_nonempty_hypothesis = bool(row["hypothesis"])
            if is_nonempty_hypothesis:
                assert row["hypothesis"] not in text


# ページのテスト（src/benchmarks/*.test.ts）が読むフィクスチャ。ページ側はこの形を前提に検査されている
PAGE_FIXTURE = REPO_ROOT / "src" / "benchmarks" / "fixture.json"
# キーがデータセット名・モデルの部品名になる（契約の項目名ではない）オブジェクト
DYNAMIC_KEY_OBJECTS = {"per_dataset", "worst_ids"}
# 中身の形がモデルごとに違ってよい項目（dtype は文字列か部品ごとの対応表。ページは読まない）
OPAQUE_KEYS = {"dtype"}
# score.py が書くが、ページ（とそのフィクスチャ）がまだ読まない項目。項目を足すだけなら schema は上げない契約なので、
# ここに挙げたものに限ってフィクスチャに無くてよい。ページが読むようになったらフィクスチャに足し、ここから消す
PAGE_UNREAD_KEYS = {".outlier_ids", ".outlier_count", ".models[].first_load_downloaded"}


def key_paths(value, path=""):
    """JSON の項目名の木を「.models[].per_dataset.*.cer_norm」のようなパスの集合にする。"""
    if isinstance(value, list):
        return set().union(*(key_paths(item, f"{path}[]") for item in value))
    if not isinstance(value, dict):
        return set()
    paths = set()
    is_dynamic = path.rsplit(".", 1)[-1] in DYNAMIC_KEY_OBJECTS
    for key, child in value.items():
        child_path = f"{path}.{'*' if is_dynamic else key}"
        paths.add(child_path)
        is_opaque = key in OPAQUE_KEYS
        if not is_opaque:
            paths |= key_paths(child, child_path)
    return paths


class TestPageContract:
    def test_page_fixture_has_exactly_the_keys_score_writes(self):
        """ページ側のフィクスチャと score.py の出力が同じ項目名の集合を持つ（ページがまだ読まない追加項目を除く）。

        Why not ページのパーサを Python から直接呼ぶ: TypeScript を pytest から動かす仕組みを増やすより、
        ページのテストが前提にしているフィクスチャと実際の出力の形を突き合わせる方が軽く、
        片方だけ項目名を変えたときにここで落ちる。
        """
        page = build_page_json(summarize({"e2b": ROWS}, METADATA), "2026-10-04T01:00:00+00:00")
        fixture = json.loads(PAGE_FIXTURE.read_text(encoding="utf-8"))
        written = key_paths(page)
        assert written >= PAGE_UNREAD_KEYS
        assert key_paths(fixture) - PAGE_UNREAD_KEYS == written - PAGE_UNREAD_KEYS


class TestLatest:
    def test_older_run_does_not_replace_latest(self, tmp_path):
        """古い run を集計し直しても latest.json は新しい run のまま。"""
        (tmp_path / "20261004T000000Z.json").write_text("{}", encoding="utf-8")
        (tmp_path / "latest.json").write_text("{}", encoding="utf-8")
        assert is_newest_scored(tmp_path, "20261004T000000Z")
        assert is_newest_scored(tmp_path, "20261005T000000Z")
        assert not is_newest_scored(tmp_path, "20261003T000000Z")

    def test_json_files_that_are_not_run_ids_are_ignored(self, tmp_path):
        """形式に合わない名前の JSON（打ち間違いの run や手で置いたファイル）は latest の判定に使わない。"""
        (tmp_path / "20261004T000000Z.json").write_text("{}", encoding="utf-8")
        (tmp_path / "zzz-typo.json").write_text("{}", encoding="utf-8")
        assert is_newest_scored(tmp_path, "20261004T000000Z")

    def test_latest_run_id_ignores_directories_that_are_not_run_ids(self, tmp_path):
        (tmp_path / "20261004T000000Z").mkdir()
        (tmp_path / "20261005T000000Z").mkdir()
        (tmp_path / "zzz-typo").mkdir()
        (tmp_path / "202610050T00000Z").mkdir()
        assert latest_run_id(tmp_path) == "20261005T000000Z"
        assert latest_run_id(tmp_path / "missing") is None


class TestMain:
    def test_writes_markdown_run_json_and_latest_with_every_model_of_the_run(self, tmp_path):
        """1 つの run_id の全モデルを 1 つの JSON にまとめ、<run_id>.json と latest.json を同じ内容で書く。"""
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "results"
        run_dir = raw_dir / "20261004T000000Z"
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(json.dumps(METADATA), encoding="utf-8")
        for model in ("e2b", "e2b-qat"):
            lines = [json.dumps({**row, "model": model}, ensure_ascii=False) for row in ROWS]
            (run_dir / f"{model}.jsonl").write_text("\n".join(lines), encoding="utf-8")

        assert main(["--raw-dir", str(raw_dir), "--out-dir", str(out_dir)]) == 0

        page = json.loads((out_dir / "20261004T000000Z.json").read_text(encoding="utf-8"))
        latest = json.loads((out_dir / "latest.json").read_text(encoding="utf-8"))
        assert page == latest
        assert [model["key"] for model in page["models"]] == ["e2b", "e4b", "e2b-qat"]
        markdown = (out_dir / "20261004T000000Z.md").read_text(encoding="utf-8")
        assert "外れ値: 0 件（なし）" in markdown
        for row in ROWS:
            assert row["reference"] not in markdown

    def test_refuses_a_malformed_run_id_without_writing_anything(self, tmp_path):
        """形式の違う run= は打ち間違いとみなし、何も書かずに止める（latest.json を奪わない）。"""
        out_dir = tmp_path / "results"
        assert main(["--run-id", "2026104T000000Z", "--out-dir", str(out_dir)]) == 2
        assert not out_dir.exists()
