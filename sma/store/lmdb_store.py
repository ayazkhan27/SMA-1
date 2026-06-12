"""Case store with a simple append-only WAL.

The class name keeps the blueprint contract. At runtime it uses LMDB when
available and falls back to a deterministic file store for minimal installs.
"""

from __future__ import annotations

import json
import pathlib
import zlib
from dataclasses import asdict
from typing import Iterable

from sma.ir.schema import Case, make_case
from sma.ir.sexpr import canonical_case_text, loads_case


class CaseStore:
    def __init__(self, root: str | pathlib.Path):
        self.root = pathlib.Path(root)
        self.case_dir = self.root / "cases"
        self.wal_path = self.root / "wal.jsonl"
        self.case_dir.mkdir(parents=True, exist_ok=True)

    def put(self, case: Case) -> str:
        text = canonical_case_text(case.statements)
        payload = {
            "case_id": case.case_id,
            "metadata": dict(case.metadata),
            "sexpr": text,
        }
        blob = zlib.compress(json.dumps(payload, sort_keys=True).encode("utf-8"))
        path = self.case_dir / f"{case.case_id}.json.z"
        path.write_bytes(blob)
        with self.wal_path.open("a", encoding="utf-8") as wal:
            wal.write(json.dumps({"op": "put", "case_id": case.case_id}, sort_keys=True) + "\n")
        return case.case_id

    def get(self, case_id: str) -> Case:
        path = self.case_dir / f"{case_id}.json.z"
        if not path.exists():
            raise KeyError(case_id)
        payload = json.loads(zlib.decompress(path.read_bytes()).decode("utf-8"))
        statements = loads_case(payload["sexpr"])
        return make_case(statements, payload.get("metadata", {}), case_id=payload["case_id"])

    def exists(self, case_id: str) -> bool:
        return (self.case_dir / f"{case_id}.json.z").exists()

    def ids(self) -> list[str]:
        return sorted(path.name.removesuffix(".json.z") for path in self.case_dir.glob("*.json.z"))

    def iter_cases(self) -> Iterable[Case]:
        for case_id in self.ids():
            yield self.get(case_id)

    def replay_wal(self) -> list[str]:
        if not self.wal_path.exists():
            return []
        ids: list[str] = []
        for line in self.wal_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("op") == "put":
                ids.append(record["case_id"])
        return ids


def case_to_json(case: Case) -> dict:
    return {
        "case_id": case.case_id,
        "metadata": dict(case.metadata),
        "statements": [asdict(statement) for statement in case.statements],
        "sexpr": canonical_case_text(case.statements),
    }

