"""metrics.py が保証すること: 正規化の範囲、編集距離、マイクロ平均、読み CER の表記ゆれ耐性。"""

import pytest

from gemma4_eval.metrics import (
    EditCount,
    edit_count,
    hiragana_to_katakana,
    levenshtein,
    micro_cer,
    normalize_norm,
    normalize_strict,
    speed,
    to_reading,
)


class TestLevenshtein:
    @pytest.mark.parametrize(
        ("reference", "hypothesis", "expected"),
        [
            ("", "", 0),
            ("abc", "abc", 0),
            ("", "abc", 3),
            ("abc", "", 3),
            ("kitten", "sitting", 3),
            ("東京都", "京都", 1),
            ("こんにちは", "こんばんは", 2),
        ],
    )
    def test_counts_character_substitutions_insertions_and_deletions(
        self, reference, hypothesis, expected
    ):
        assert levenshtein(reference, hypothesis) == expected

    def test_is_symmetric(self):
        assert levenshtein("音声認識", "音声人識別") == levenshtein("音声人識別", "音声認識")

    def test_handles_long_runaway_hypothesis(self):
        """モデルが同じ句を繰り返して仮説が極端に長くなっても、超過分がそのまま挿入として数えられる。"""
        assert levenshtein("はい", "はい" * 500) == 998


class TestNormalizeStrict:
    def test_removes_only_whitespace_and_newlines(self):
        assert normalize_strict(" 今日は\n 晴れ　です。\t") == "今日は晴れです。"

    def test_keeps_punctuation_and_width_differences_as_errors(self):
        """全角/半角や句読点の違いは strict では別の文字のまま残る。"""
        assert normalize_strict("ＡＢＣ、") != normalize_strict("ABC")

    def test_applies_nfc_so_decomposed_kana_matches_composed(self):
        decomposed = "が"  # か + 結合用濁点
        assert normalize_strict(decomposed) == "が"


class TestNormalizeNorm:
    def test_removes_punctuation_symbols_and_whitespace(self):
        assert normalize_norm("「はい、そうです！」 …… ♪") == "はいそうです"

    def test_folds_width_and_case(self):
        assert normalize_norm("ＧｅｍｍａＡＢＣ１２３ ｶﾀｶﾅ") == "gemmaabc123カタカナ"

    def test_keeps_long_vowel_mark(self):
        """長音記号「ー」は記号ではなく文字（Lm）なので消さない。"""
        assert normalize_norm("コーヒー。") == "コーヒー"

    def test_drops_trailing_period_noise_in_common_voice_references(self):
        """Common Voice の参照末尾の「。.」は norm では誤りに数えない。"""
        assert normalize_norm("先生です。.") == normalize_norm("先生です")


class TestReading:
    def test_converts_kanji_to_katakana_reading(self):
        assert to_reading("東京に行きました") == "トウキョウニイキマシタ"

    def test_ignores_kanji_choice_and_okurigana(self):
        """送り仮名の違い（取り扱い/取扱い）やかな書きは、読みが同じなら誤りにならない。"""
        assert to_reading("取り扱い") == to_reading("取扱い")
        assert to_reading("ちょっと見ていきます") == to_reading("チョット見て行きます")

    def test_falls_back_to_normalized_surface_when_dictionary_has_no_reading(self):
        """数字や未知の英単語のように辞書に読みが無い形態素は、norm と同じ表記（半角・小文字）で残す。"""
        assert to_reading("xyzzyで2026年") == "xyzzyデ2026ネン"
        assert to_reading("ＸＹＺＺＹで２０２６年") == "xyzzyデ2026ネン"

    @pytest.mark.parametrize("spelled", ["エヌエイチケー", "ＮＨＫ", "nhk"])
    def test_latin_acronym_reads_the_same_as_its_spelling(self, spelled):
        """半角の頭字語「NHK」も辞書の読み（エヌエイチケー）になり、カナ書き・全角表記と誤りにならない。"""
        assert to_reading("NHKのニュース") == to_reading(f"{spelled}のニュース")
        assert edit_count("NHK", spelled, "cer_reading").edits == 0

    def test_empty_after_normalization_is_empty(self):
        assert to_reading("、。！") == ""

    def test_hiragana_to_katakana_leaves_other_characters(self):
        assert hiragana_to_katakana("ぁあゖー漢A") == "ァアヶー漢A"


class TestEditCount:
    def test_reading_cer_is_zero_for_kanji_variation_but_norm_is_not(self):
        reference, hypothesis = "取り扱い説明書", "取扱い説明書"
        assert edit_count(reference, hypothesis, "cer_norm").edits == 1
        assert edit_count(reference, hypothesis, "cer_reading").edits == 0

    def test_strict_counts_punctuation_that_norm_ignores(self):
        reference, hypothesis = "はい、そうです。", "はいそうです"
        assert edit_count(reference, hypothesis, "cer_strict") == EditCount(2, 8)
        assert edit_count(reference, hypothesis, "cer_norm") == EditCount(0, 6)

    def test_reference_length_is_measured_after_normalization(self):
        assert edit_count("ＡＢ。", "ab", "cer_norm").reference_chars == 2


class TestMicroCer:
    def test_divides_total_edits_by_total_reference_characters(self):
        """マイクロ平均: (1 + 0) / (2 + 8)。クリップごとの CER の平均（0.25）ではない。"""
        pairs = [("ab", "aX"), ("abcdefgh", "abcdefgh")]
        total = micro_cer(pairs, "cer_strict")
        assert total == EditCount(1, 10)
        assert total.rate == pytest.approx(0.1)

    def test_empty_reference_contributes_insertions_only(self):
        """参照が空のクリップは分母を増やさず、仮説の文字数だけ分子に足す。"""
        total = micro_cer([("", "あ"), ("あいう", "あいう")], "cer_norm")
        assert total == EditCount(1, 3)

    def test_rate_is_none_when_every_reference_is_empty(self):
        """参照文字数が 0 なら CER は定義できないので None（0 や無限大で平均を歪めない）。"""
        assert micro_cer([("", "あ"), ("。", "")], "cer_norm").rate is None
        assert micro_cer([], "cer_norm").rate is None

    def test_rate_can_exceed_one_with_insertions(self):
        assert micro_cer([("あ", "あいうえ")], "cer_norm").rate == pytest.approx(3.0)


class TestSpeed:
    def test_rtf_is_total_inference_time_over_total_audio_time(self):
        result = speed([(10.0, 100.0, 2_000.0), (30.0, 300.0, 6_000.0)])
        assert result.rtf == pytest.approx(8_000 / 40 / 1000)
        assert result.rtf_with_decode == pytest.approx(8_400 / 40 / 1000)
        assert result.audio_seconds == pytest.approx(40.0)

    def test_rtf_is_none_without_audio(self):
        result = speed([])
        assert result.rtf is None
        assert result.rtf_with_decode is None
