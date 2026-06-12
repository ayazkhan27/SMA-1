"""Structured JSON/CSV/XML-ish Tier-0 encoder."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Mapping

from sma.ir.schema import Statement, entity, make_case, stmt

from .base import EncodeResult


class StructuredEncoder:
    adapter_id = "structured"
    version = "0.1.0"

    def encode(self, artifact: str, **kwargs) -> EncodeResult:
        fmt = kwargs.get("format") or infer_format(artifact)
        statements: list[Statement] = []
        if fmt == "json":
            data = json.loads(artifact)
            encode_json(data, statements, "root")
        else:
            reader = csv.DictReader(io.StringIO(artifact))
            for i, row in enumerate(reader):
                row_ent = entity(f"row_{i}", "row")
                for key, value in row.items():
                    if value is not None and value != "":
                        statements.append(stmt(key, row_ent, entity(value, "value")))
        case = make_case(statements or [stmt("emptyStructured", entity("root"))], {"adapter": self.adapter_id, "tier": 0})
        return EncodeResult(case, ())


def infer_format(artifact: str) -> str:
    stripped = artifact.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        return "json"
    return "csv"


def encode_json(value, statements: list[Statement], path: str) -> None:
    subject = entity(path, "json_node")
    if isinstance(value, Mapping):
        for key, child in sorted(value.items()):
            child_path = f"{path}.{key}"
            statements.append(stmt(str(key), subject, entity(child_path, "json_node")))
            encode_json(child, statements, child_path)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            child_path = f"{path}.{i}"
            statements.append(stmt("item", subject, entity(child_path, "json_node")))
            encode_json(child, statements, child_path)
    else:
        statements.append(stmt("value", subject, entity(str(value), "value")))

