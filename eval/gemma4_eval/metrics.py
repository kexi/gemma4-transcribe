"""文字起こし結果の指標（正規化・編集距離・CER・読み CER・速度）。すべて純粋関数。

集計はマイクロ平均（総編集数 / 参照総文字数）。クリップごとの CER を平均するマクロ平均は、
数文字しかない短いクリップ 1 件の誤りが長いクリップと同じ重みになり、データセット間で比較できなくなるため採らない。
"""

from __future__ import annotations

import os
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import cache

# 指標名 → 正規化関数。score.py の列順もこの順に従う
METRIC_NAMES: tuple[str, ...] = ("cer_strict", "cer_norm", "cer_reading")

_HIRAGANA_START = ord("ぁ")
_HIRAGANA_END = ord("ゖ")
_KATAKANA_OFFSET = ord("ァ") - ord("ぁ")


def levenshtein(reference: str, hypothesis: str) -> int:
    """文字単位の編集距離（置換・挿入・削除がそれぞれ 1）。

    2 行分の DP だけを持つ。Gemma が同じ句を繰り返し続けると仮説が数千文字になることがあり、
    全行列を持つと参照長 × 仮説長のメモリになるため。
    """
    if reference == hypothesis:
        return 0
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)

    previous = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, start=1):
        current = [i]
        for j, hyp_char in enumerate(hypothesis, start=1):
            is_same = ref_char == hyp_char
            substitution = previous[j - 1] + (0 if is_same else 1)
            deletion = previous[j] + 1
            insertion = current[j - 1] + 1
            current.append(min(substitution, deletion, insertion))
        previous = current
    return previous[-1]


def _remove_whitespace(text: str) -> str:
    # str.split() は全角スペース（U+3000）や改行も空白として扱う
    return "".join(text.split())


def normalize_strict(text: str) -> str:
    """NFC と空白・改行の除去だけ。表記ゆれ（全角半角・句読点・漢字/かな）はすべて誤りとして残す。"""
    return _remove_whitespace(unicodedata.normalize("NFC", text))


def _is_punctuation_or_symbol(char: str) -> bool:
    category = unicodedata.category(char)
    return category.startswith(("P", "S"))


def normalize_norm(text: str) -> str:
    """NFKC・英字小文字化・句読点と記号（Unicode カテゴリ P*, S*）と空白の除去。

    一般的な日本語 ASR 評価に近い正規化。長音記号「ー」は Lm（修飾文字）なので残る。
    """
    folded = unicodedata.normalize("NFKC", text).lower()
    kept = (char for char in folded if not _is_punctuation_or_symbol(char))
    return _remove_whitespace("".join(kept))


def hiragana_to_katakana(text: str) -> str:
    """ひらがなをカタカナに寄せる。それ以外の文字はそのまま。"""
    chars = []
    for char in text:
        code = ord(char)
        is_hiragana = _HIRAGANA_START <= code <= _HIRAGANA_END
        chars.append(chr(code + _KATAKANA_OFFSET) if is_hiragana else char)
    return "".join(chars)


@cache
def _tagger():
    # import を遅らせるのは、読み CER を使わない処理（prepare など）で MeCab 辞書の読み込みを待たないため
    import fugashi
    import unidic_lite

    # Why not fugashi.Tagger() の自動検出: 環境に unidic（フル版）が入っているとそちらが選ばれ、
    # 辞書の版で読みが変わって過去の結果と比較できなくなるため、unidic-lite を明示する
    dicdir = unidic_lite.DICDIR
    mecabrc = os.path.join(dicdir, "mecabrc")
    return fugashi.Tagger(f'-d "{dicdir}" -r "{mecabrc}"')


def _is_missing_reading(kana: str | None) -> bool:
    return kana is None or kana in ("", "*")


# 半角英数字 → 全角大文字。unidic は「ＮＨＫ」「ＤＮＡ」のような全角大文字の頭字語にだけ読みを持ち、
# normalize_norm が作る半角小文字の「nhk」は未知語になって読みが付かないため、解析用にだけ書き換える
_FULLWIDTH_UPPER = str.maketrans(
    {
        **{chr(code): chr(code - ord("a") + ord("Ａ")) for code in range(ord("a"), ord("z") + 1)},
        **{chr(code): chr(code - ord("A") + ord("Ａ")) for code in range(ord("A"), ord("Z") + 1)},
        **{chr(code): chr(code - ord("0") + ord("０")) for code in range(ord("0"), ord("9") + 1)},
    }
)


def to_reading(text: str) -> str:
    """normalize_norm 後の文字列を形態素解析し、各形態素の読み（kana）をカタカナで連結する。

    解析には英数字を全角大文字にした文字列を渡す（「NHK」と「エヌエイチケー」を同じ読みにするため）。
    辞書に読みが無い形態素（数字・未知の英単語）は、表層を NFKC + 小文字に戻して使う（norm と同じ表記）。
    漢字の選び方や送り仮名の違い（「取り扱い」と「取扱い」）を無視した「音の正しさ」を測るための変換。
    """
    normalized = normalize_norm(text)
    if not normalized:
        return ""
    parts = []
    for word in _tagger()(normalized.translate(_FULLWIDTH_UPPER)):
        kana = getattr(word.feature, "kana", None)
        is_missing = _is_missing_reading(kana)
        parts.append(unicodedata.normalize("NFKC", word.surface).lower() if is_missing else kana)
    return hiragana_to_katakana("".join(parts))


NORMALIZERS: dict[str, Callable[[str], str]] = {
    "cer_strict": normalize_strict,
    "cer_norm": normalize_norm,
    "cer_reading": to_reading,
}


@dataclass(frozen=True)
class EditCount:
    """1 クリップ（または集計）の編集数と参照文字数。CER はこの比で、合算してから割る。"""

    edits: int
    reference_chars: int

    def __add__(self, other: EditCount) -> EditCount:
        return EditCount(self.edits + other.edits, self.reference_chars + other.reference_chars)

    @property
    def rate(self) -> float | None:
        """編集数 / 参照文字数。参照が空のときは定義できないので None（0 や inf にすると平均を歪める）。"""
        if self.reference_chars == 0:
            return None
        return self.edits / self.reference_chars


def edit_count(reference: str, hypothesis: str, metric: str) -> EditCount:
    """指標 metric の正規化を両辺にかけてから編集距離を数える。"""
    normalize = NORMALIZERS[metric]
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    return EditCount(levenshtein(ref, hyp), len(ref))


def micro_cer(pairs: Iterable[tuple[str, str]], metric: str) -> EditCount:
    """(参照, 仮説) の組を総編集数 / 参照総文字数でまとめる。参照が空の組は分子にだけ効く。"""
    total = EditCount(0, 0)
    for reference, hypothesis in pairs:
        total = total + edit_count(reference, hypothesis, metric)
    return total


@dataclass(frozen=True)
class Speed:
    """速度の集計。RTF は 1 未満なら実時間より速い。"""

    rtf: float | None
    rtf_with_decode: float | None
    audio_seconds: float


def speed(clips: Sequence[tuple[float, float, float]]) -> Speed:
    """(duration_s, decode_ms, infer_ms) の列から RTF = Σinfer_ms / Σduration_s / 1000 を出す。"""
    audio_seconds = sum(duration for duration, _, _ in clips)
    if audio_seconds <= 0:
        return Speed(None, None, 0.0)
    infer_ms = sum(infer for _, _, infer in clips)
    decode_ms = sum(decode for _, decode, _ in clips)
    return Speed(
        rtf=infer_ms / audio_seconds / 1000,
        rtf_with_decode=(infer_ms + decode_ms) / audio_seconds / 1000,
        audio_seconds=audio_seconds,
    )
