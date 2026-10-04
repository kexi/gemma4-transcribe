"""run_id（UTC のタイムスタンプ `YYYYMMDDTHHMMSSZ`）の形式。run_browser と score で同じ規則を使う。

Why not 任意の文字列を run_id として受け付ける: `run=` の打ち間違い（`run=2026100T...`）が黙って新しい run になり、
しかも名前順で最新の run として latest.json を奪ってしまうため。形式を固定し、名前順 = 時刻順を保証する
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import UTC, datetime

RUN_ID_FORMAT = "%Y%m%dT%H%M%SZ"
RUN_ID_PATTERN = re.compile(r"^\d{8}T\d{6}Z$")


def is_valid_run_id(name: str) -> bool:
    return RUN_ID_PATTERN.fullmatch(name) is not None


def new_run_id(now: datetime | None = None) -> str:
    return (now or datetime.now(UTC)).strftime(RUN_ID_FORMAT)


def run_ids(names: Iterable[str]) -> list[str]:
    """run_id の形式に合う名前だけを名前順（= 時刻順）で返す。手で置いたディレクトリや latest などは無視する。"""
    return sorted(name for name in names if is_valid_run_id(name))
