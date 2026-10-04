"""1 イベント 1 行の JSON ログ（stderr）。grep / jq で追えるようにフィールド名を揃える。"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from typing import Any


def log(event: str, **fields: Any) -> None:
    """event と ts を必ず含む 1 行の JSON を stderr に出す。

    stdout ではなく stderr に出すのは、stdout を結果のパイプ（jq 等）に使えるよう空けておくため。
    """
    record = {"ts": datetime.now(UTC).isoformat(timespec="milliseconds"), "event": event}
    record.update(fields)
    print(json.dumps(record, ensure_ascii=False, default=str), file=sys.stderr, flush=True)
