"""WAL helpers."""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass


@dataclass(frozen=True)
class WalRecord:
    op: str
    case_id: str


def read_wal(path: str | pathlib.Path) -> list[WalRecord]:
    p = pathlib.Path(path)
    if not p.exists():
        return []
    records: list[WalRecord] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            data = json.loads(line)
            records.append(WalRecord(op=data["op"], case_id=data["case_id"]))
    return records

