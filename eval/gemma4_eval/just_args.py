"""just のレシピ引数で `name=value` と書かれたものを、その名前の引数に振り直す。

just のレシピ引数は位置引数だけなので、`just eval-run models=e2b,e4b` と書くと 1 番目の引数（models）に
文字列 "models=e2b,e4b" が入り、`just eval-run run=<run_id>` でも 1 番目に "run=<run_id>" が入る。
Why not justfile 側で `*args` を受けて分解する: `just --list` に引数名と既定値が出なくなり、
`just eval-run e2b 3` のような位置引数の書き方と両立させる処理をシェルで書くことになるため
"""

from __future__ import annotations

from collections.abc import Mapping


def reassign_named(values: Mapping[str, str]) -> dict[str, str]:
    """{引数名: 渡された文字列} を受け取り、`name=value` 形式の値を name の引数へ移した辞書を返す。

    name が values のキーに無いもの（"=" を含む普通の値）は位置引数のまま残す。
    位置引数で埋まっている引数に同じ名前の `name=value` も渡されたら、どちらを使うか決められないので ValueError。
    """
    positional: dict[str, str] = {}
    named: dict[str, str] = {}
    for slot, value in values.items():
        name, separator, rest = value.partition("=")
        is_named = bool(separator) and name in values
        if not is_named:
            positional[slot] = value
            continue
        if name in named:
            raise ValueError(f"{name}= が 2 回指定されています")
        named[name] = rest
    for name, value in named.items():
        is_conflict = positional.get(name, "").strip() != ""
        if is_conflict:
            raise ValueError(
                f"{name} が位置引数（{positional[name]}）と {name}={value} の両方で指定されています"
            )
        positional[name] = value
    return {name: positional.get(name, "") for name in values}
