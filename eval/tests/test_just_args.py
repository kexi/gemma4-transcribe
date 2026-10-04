"""just_args.py が保証すること: `just eval-run models=a,b` のような name=value 形式が、位置引数の書き方と同じ意味になる。"""

import pytest

from gemma4_eval.just_args import reassign_named


class TestReassignNamed:
    def test_positional_values_are_kept(self):
        assert reassign_named({"models": "e2b", "limit": "3", "run": ""}) == {
            "models": "e2b",
            "limit": "3",
            "run": "",
        }

    def test_named_value_in_the_first_slot_moves_to_its_own_argument(self):
        """`just eval-run models=e2b,e4b` は 1 番目に "models=e2b,e4b" が入る。複数モデルを 1 回で測れる。"""
        assert reassign_named(
            {"models": "models=e2b,e2b-qat,e4b,e4b-qat", "limit": "", "run": ""}
        ) == {
            "models": "e2b,e2b-qat,e4b,e4b-qat",
            "limit": "",
            "run": "",
        }

    def test_named_run_leaves_models_empty_for_the_default(self):
        """`just eval-run run=<id>` は models を空にする（呼び出し側が既定モデルで埋める）。"""
        assert reassign_named({"models": "run=20261004T000000Z", "limit": "", "run": ""}) == {
            "models": "",
            "limit": "",
            "run": "20261004T000000Z",
        }

    def test_positional_and_named_can_be_mixed(self):
        assert reassign_named({"models": "e2b", "limit": "run=X", "run": ""}) == {
            "models": "e2b",
            "limit": "",
            "run": "X",
        }

    def test_value_with_unknown_name_is_left_as_a_positional_value(self):
        assert reassign_named({"run": "a=b"}) == {"run": "a=b"}

    def test_conflicting_positional_and_named_value_is_an_error(self):
        """どちらを使うか決められない指定は黙って片方を捨てない。"""
        with pytest.raises(ValueError, match="models"):
            reassign_named({"models": "e2b", "limit": "models=e4b", "run": ""})

    def test_same_name_twice_is_an_error(self):
        with pytest.raises(ValueError, match="limit"):
            reassign_named({"models": "limit=1", "limit": "limit=2", "run": ""})
