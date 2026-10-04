"""run_browser.py の純粋な部分が保証すること: raw 行の形、評価範囲の切り出し、途中再開（ブラウザ不要）、
run_id の検証、再開時に最初の実行環境を残すこと、初回読み込みでダウンロードが起きたかの判定。"""

import json
import urllib.error
import urllib.request
from datetime import UTC, datetime

import pytest

from gemma4_eval.run_browser import (
    CRASHES_BEFORE_FAILURE,
    ModelOutcome,
    ModelRun,
    SessionOutcome,
    UnplannedClipError,
    chrome_version,
    detect_download,
    eval_url,
    parse_limit,
    plan_items,
    planned_datasets,
    planned_ids_digest,
    prefetch_download,
    raw_row,
    read_recorded_ids,
    record_chrome,
    record_first_load,
    resolve_run_id,
    resolve_settings,
    start_server_near,
    to_snake_case,
)
from gemma4_eval.server import start_server


class TestRawRow:
    def test_converts_eval_result_to_snake_case_and_adds_session_fields(self):
        result = {
            "id": "jsut-000001",
            "dataset": "jsut",
            "reference": "r",
            "hypothesis": "h",
            "durationS": 1.5,
            "decodeMs": 3.0,
            "inferMs": 900.0,
            "segments": 1,
            "error": None,
        }
        row = raw_row(result, "e2b", "run-1", 1234.0, 567.0)
        assert row == {
            "id": "jsut-000001",
            "dataset": "jsut",
            "reference": "r",
            "hypothesis": "h",
            "duration_s": 1.5,
            "decode_ms": 3.0,
            "infer_ms": 900.0,
            "segments": 1,
            "error": None,
            "model": "e2b",
            "run_id": "run-1",
            "load_ms": 1234.0,
            "warmup_ms": 567.0,
        }

    @pytest.mark.parametrize(
        ("camel", "snake"), [("durationS", "duration_s"), ("loadMs", "load_ms"), ("id", "id")]
    )
    def test_snake_case(self, camel, snake):
        assert to_snake_case(camel) == snake


class TestPlanItems:
    def test_matches_eval_page_offset_and_limit(self):
        """評価ページの selectItems（items.slice(offset, offset + limit)）と同じ範囲を選ぶ。"""
        items = list(range(10))
        assert plan_items(items, 2, 3) == [2, 3, 4]
        assert plan_items(items, 8, None) == [8, 9]
        assert plan_items(items, 8, 5) == [8, 9]

    def test_empty_limit_means_unlimited(self):
        assert parse_limit("") is None
        assert parse_limit(" 5 ") == 5


class TestResume:
    def test_pending_index_skips_recorded_clips_and_record_ignores_duplicates(self, tmp_path):
        """同じ run_id で再実行すると、記録済みのクリップを飛ばして続きから測り、同じ行を二重に書かない。"""
        out_path = tmp_path / "e2b.jsonl"
        out_path.write_text(json.dumps({"id": "a"}) + "\n" + '{"id": "b', encoding="utf-8")
        model_run = ModelRun(
            model="e2b",
            planned=[{"id": "a"}, {"id": "b"}, {"id": "c"}],
            out_path=out_path,
            run_id="run-1",
            recorded=read_recorded_ids(out_path),
        )
        assert model_run.pending_index() == 1
        assert model_run.record({"id": "a"}) is False
        assert model_run.record({"id": "b"}) is True
        assert model_run.pending_index() == 2

    def test_record_rejects_clip_outside_the_plan_without_writing(self, tmp_path):
        """別の manifest や範囲の結果が届いたら、run に混ぜずに止める。"""
        out_path = tmp_path / "e2b.jsonl"
        model_run = ModelRun("e2b", [{"id": "a"}], out_path, "run-1")
        with pytest.raises(UnplannedClipError, match="zzz"):
            model_run.record({"id": "zzz"})
        assert not out_path.exists()


class TestResolveSettings:
    def test_new_run_uses_requested_values_or_defaults(self):
        assert resolve_settings(None, {"offset": None, "limit": 3, "language": None}) == {
            "offset": 0,
            "limit": 3,
            "language": "Japanese",
        }

    def test_resume_reuses_recorded_range_when_not_specified(self):
        """--run-id だけで再開しても、最初の実行の offset / limit / 言語のまま続きを測る（全件に広がらない）。"""
        existing = {"offset": 10, "limit": 5, "language": "Japanese"}
        resolved = resolve_settings(existing, {"offset": None, "limit": None, "language": None})
        assert resolved == existing

    def test_resume_accepts_the_same_explicit_values(self):
        existing = {"offset": 0, "limit": 5, "language": "Japanese"}
        assert resolve_settings(existing, {"offset": 0, "limit": 5, "language": "Japanese"}) == (
            existing
        )

    @pytest.mark.parametrize(
        ("requested", "name"),
        [
            ({"limit": 10}, "--limit"),
            ({"offset": 3}, "--offset"),
            ({"language": "English"}, "--language"),
        ],
    )
    def test_resume_refuses_a_different_range_or_language(self, requested, name):
        """同じ run_id に別条件の結果を混ぜない。"""
        existing = {"offset": 0, "limit": 5, "language": "Japanese"}
        with pytest.raises(ValueError, match=name):
            resolve_settings(existing, requested)

    def test_unlimited_run_refuses_a_new_limit(self):
        """全件（limit=None）で始めた run を件数指定で再開しようとしたら止める。"""
        with pytest.raises(ValueError, match="--limit"):
            resolve_settings({"offset": 0, "limit": None}, {"limit": 3})


class TestPlannedSummary:
    def test_digest_changes_when_the_planned_ids_change(self):
        """manifest を作り直して ID の並びが変わったことを、再開前に検出できる。"""
        first = [{"id": "a"}, {"id": "b"}]
        assert planned_ids_digest(first) == planned_ids_digest([{"id": "a"}, {"id": "b"}])
        assert planned_ids_digest(first) != planned_ids_digest([{"id": "b"}, {"id": "a"}])

    def test_counts_and_audio_seconds_per_dataset(self):
        planned = [
            {"id": "j1", "dataset": "jsut", "duration_s": 1.5},
            {"id": "r1", "dataset": "reazon", "duration_s": 2.0},
            {"id": "j2", "dataset": "jsut", "duration_s": 2.25},
        ]
        assert planned_datasets(planned) == {
            "jsut": {"n": 2, "audio_s": 3.75},
            "reazon": {"n": 1, "audio_s": 2.0},
        }


def _item(clip_id):
    return {"id": clip_id, "dataset": "jsut", "reference": "r", "duration_s": 1.0}


class TestStuckClip:
    def make_run(self, tmp_path, recorded=()):
        return ModelRun(
            "e2b",
            [_item("a"), _item("b"), _item("c")],
            tmp_path / "e2b.jsonl",
            "run-1",
            recorded=set(recorded),
        )

    def test_timeout_is_recorded_against_the_clip_in_progress(self, tmp_path):
        """タイムアウトは、未記録の先頭（a）ではなく処理中だったクリップ（b）の失敗として記録する。"""
        model_run = self.make_run(tmp_path)
        outcome = SessionOutcome(
            status="clip-timeout", load_ms=1.0, warmup_ms=2.0, stalled_ms=300_000, current_id="b"
        )
        row = model_run.stuck_row_for(outcome)
        assert row["id"] == "b"
        assert row["error"].startswith("timeout")
        assert row["warmup_ms"] == 2.0

    def test_crash_is_not_blamed_on_a_completed_but_unread_clip(self, tmp_path):
        """クラッシュ 1 回目は記録しない（直前に終わった結果を読めていないだけかもしれない）。

        同じクリップで CRASHES_BEFORE_FAILURE 回落ちたら、そのクリップの失敗として記録する。
        """
        model_run = self.make_run(tmp_path, recorded={"a"})
        outcome = SessionOutcome(status="crashed", load_ms=1.0, current_id="b", message="boom")
        for _ in range(CRASHES_BEFORE_FAILURE - 1):
            assert model_run.stuck_row_for(outcome) is None
        row = model_run.stuck_row_for(outcome)
        assert row["id"] == "b"
        assert row["error"].startswith("crashed")

    def test_crashes_on_different_clips_are_counted_separately(self, tmp_path):
        model_run = self.make_run(tmp_path)
        assert (
            model_run.stuck_row_for(SessionOutcome("crashed", load_ms=1.0, current_id="a")) is None
        )
        assert (
            model_run.stuck_row_for(SessionOutcome("crashed", load_ms=1.0, current_id="b")) is None
        )

    @pytest.mark.parametrize(
        "outcome",
        [
            # 読み込み前のクラッシュはクリップのせいではない
            SessionOutcome("crashed", load_ms=None, current_id="a"),
            # クリップの合間（currentId が null）
            SessionOutcome("clip-timeout", load_ms=1.0, current_id=None),
            # 予定外の ID
            SessionOutcome("clip-timeout", load_ms=1.0, current_id="zzz"),
            # 結果がすでに記録済み
            SessionOutcome("clip-timeout", load_ms=1.0, current_id="c"),
            # page-error はページ側がそのクリップの結果を記録済み
            SessionOutcome("page-error", load_ms=1.0, current_id="b"),
        ],
    )
    def test_records_nothing_when_no_clip_is_to_blame(self, tmp_path, outcome):
        model_run = self.make_run(tmp_path, recorded={"c"})
        for _ in range(CRASHES_BEFORE_FAILURE):
            assert model_run.stuck_row_for(outcome) is None


class TestEvalUrl:
    def test_points_at_the_given_manifest_under_data(self):
        """--manifest で別名の manifest を渡したら、ページもその manifest を読む。"""
        url = eval_url("http://x", "e2b", 0, None, "Japanese", "small.json")
        assert "manifest=%2Fdata%2Fsmall.json" in url

    def test_limit_zero_is_kept_for_the_prefetch_session(self):
        """ダウンロード専用のセッションは limit=0 を明示する（省略すると全件を測ってしまう）。"""
        assert "limit=0" in eval_url("http://x", "e2b", 0, 0, "Japanese")

    def test_contains_contract_parameters(self):
        url = eval_url("http://127.0.0.1:8765", "e2b", 3, 5, "Japanese")
        assert url == (
            "http://127.0.0.1:8765/eval.html?model=e2b&manifest=%2Fdata%2Fmanifest.json"
            "&offset=3&language=Japanese&limit=5"
        )

    def test_omits_limit_when_unlimited(self):
        assert "limit" not in eval_url("http://x", "e4b", 0, None, "Japanese")


class TestChromeVersion:
    def test_reads_full_version_from_brand_list(self):
        probe = {
            "fullVersionList": [
                {"brand": "Not=A?Brand", "version": "99.0.0.0"},
                {"brand": "Google Chrome", "version": "140.0.7339.80"},
            ]
        }
        assert chrome_version(probe) == "140.0.7339.80"

    def test_unknown_without_brand_list(self):
        assert chrome_version({"fullVersionList": None}) is None


class TestServer:
    def test_serves_dist_at_root_and_data_under_data_prefix(self, tmp_path):
        """1 つのオリジンで / → dist-eval、/data/ → eval/data を返し、data の外には出られない。"""
        dist = tmp_path / "dist-eval"
        data = tmp_path / "data"
        dist.mkdir()
        data.mkdir()
        (dist / "eval.html").write_text("page", encoding="utf-8")
        (dist / "worker.js").write_text("js", encoding="utf-8")
        (data / "manifest.json").write_text("{}", encoding="utf-8")
        (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")

        server = start_server(dist, data, port=0)
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urllib.request.urlopen(f"{base}/eval.html?model=e2b") as response:
                assert response.read() == b"page"
            with urllib.request.urlopen(f"{base}/worker.js") as response:
                assert response.headers["Content-Type"] == "text/javascript"
            with urllib.request.urlopen(f"{base}/data/manifest.json") as response:
                assert response.read() == b"{}"
                assert response.headers["Content-Type"] == "application/json"
            with pytest.raises(urllib.error.HTTPError):
                urllib.request.urlopen(f"{base}/data/../secret.txt")
            with pytest.raises(urllib.error.HTTPError):
                urllib.request.urlopen(f"{base}/data/%2e%2e/secret.txt")
        finally:
            server.shutdown()


class TestStartServerNear:
    def test_uses_the_next_port_when_the_preferred_one_is_taken(self, tmp_path):
        """既定ポートが塞がっていても空きポートへは逃げず、隣のポートを使う（モデルキャッシュのオリジンを保つため）。"""
        blocker = start_server(tmp_path, tmp_path, port=0)
        preferred = blocker.server_port
        try:
            server = start_server_near(tmp_path, tmp_path, preferred, attempts=10)
            try:
                assert preferred < server.server_port < preferred + 10
            finally:
                server.shutdown()
        finally:
            blocker.shutdown()

    def test_falls_back_to_an_os_chosen_port_when_every_candidate_is_taken(self, tmp_path):
        """候補がすべて塞がっていても評価は止めず、OS が選んだポートで起動する。"""
        blocker = start_server(tmp_path, tmp_path, port=0)
        try:
            server = start_server_near(tmp_path, tmp_path, blocker.server_port, attempts=1)
            try:
                assert server.server_port != blocker.server_port
            finally:
                server.shutdown()
        finally:
            blocker.shutdown()


class TestResolveRunId:
    def test_empty_run_id_starts_a_new_run_named_by_utc_time(self, tmp_path):
        now = datetime(2026, 10, 4, 2, 56, 47, tzinfo=UTC)
        assert resolve_run_id(" ", tmp_path, now) == "20261004T025647Z"

    def test_existing_run_with_run_json_is_resumed(self, tmp_path):
        (tmp_path / "20261004T025647Z").mkdir()
        (tmp_path / "20261004T025647Z" / "run.json").write_text("{}", encoding="utf-8")
        assert resolve_run_id("20261004T025647Z", tmp_path) == "20261004T025647Z"

    @pytest.mark.parametrize("typo", ["2026104T025647Z", "20261004T025647", "latest", "../x"])
    def test_malformed_run_id_is_refused(self, tmp_path, typo):
        """打ち間違いの run= を新しい run として始めない（名前順で latest.json を奪わないため）。"""
        with pytest.raises(ValueError, match="YYYYMMDDTHHMMSSZ"):
            resolve_run_id(typo, tmp_path)

    def test_explicit_run_id_without_run_json_is_refused(self, tmp_path):
        """形式が正しくても、存在しない run を指定したら再開ではないので止める（ディレクトリだけあっても同じ）。"""
        with pytest.raises(ValueError, match="見つかりません"):
            resolve_run_id("20261004T025647Z", tmp_path)
        (tmp_path / "20261004T025647Z").mkdir()
        with pytest.raises(ValueError, match="見つかりません"):
            resolve_run_id("20261004T025647Z", tmp_path)


CHROME_FIRST = {"version": "154.0.1.2", "gpu": {"vendor": "apple"}}
CHROME_LATER = {"version": "155.0.0.1", "gpu": {"vendor": "apple"}}


class TestRecordChrome:
    def test_new_run_records_the_browser(self):
        metadata = {}
        record_chrome(metadata, CHROME_FIRST, is_resume=False)
        assert metadata["chrome"] == CHROME_FIRST

    def test_resume_keeps_the_original_browser_and_stacks_the_new_one_on_the_resume(self):
        """再開しても最初の実行の chrome は書き換えず、再開時の Chrome は resumes の最後に残す。"""
        metadata = {
            "chrome": CHROME_FIRST,
            "resumes": [{"started_at": "a", "chrome": CHROME_FIRST}, {"started_at": "b"}],
        }
        record_chrome(metadata, CHROME_LATER, is_resume=True)
        assert metadata["chrome"] == CHROME_FIRST
        assert metadata["resumes"][-1]["chrome"] == CHROME_LATER
        assert metadata["resumes"][0]["chrome"] == CHROME_FIRST


class TestFirstLoad:
    @pytest.mark.parametrize(
        ("before", "after", "expected"),
        [
            (0, 12, True),  # 空のキャッシュに全ファイルを保存した
            (5, 12, True),  # 途中まで保存済みで、残りをダウンロードした
            (12, 12, False),  # すべてキャッシュから読んだ
            (0, 0, None),  # 読み込み後も 0 件: Cache API が使われておらず判断できない
            (None, 12, None),
            (12, None, None),
        ],
    )
    def test_detects_download_from_cached_file_counts(self, before, after, expected):
        assert detect_download(before, after) is expected

    def test_download_in_a_crashed_prefetch_attempt_is_not_missed(self):
        """1 回目がダウンロード途中で落ち、2 回目は残りをキャッシュから読んだだけでも、ダウンロードありとする。"""
        attempts = [
            SessionOutcome("crashed", cached_files_before=0, cached_files_after=None),
            SessionOutcome("done", load_ms=1.0, cached_files_before=12, cached_files_after=12),
        ]
        assert prefetch_download(attempts) is True
        assert prefetch_download([]) is None

    def test_keeps_the_first_measured_load_together_with_its_download_flag(self):
        """再開時（キャッシュからの読み込み）の値で、最初に測れた初回読み込みとダウンロードの有無を上書きしない。"""
        model_metadata = {}
        record_first_load(model_metadata, ModelOutcome("done", 90_000.0, True))
        record_first_load(model_metadata, ModelOutcome("done", 5_000.0, False))
        assert model_metadata == {"first_load_ms": 90_000.0, "first_load_downloaded": True}
