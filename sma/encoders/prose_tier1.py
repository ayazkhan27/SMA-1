"""Flagged Tier-1 prose encoder.

This is a deterministic connective/clause fallback. It is marked Tier-1 and
does not support headline Tier-0 claims.
"""

from __future__ import annotations

import re
from hashlib import blake2s

from sma.ir.schema import Statement, entity, make_case, stmt

from .base import EncodeResult


CONNECTIVES = {
    "because": "cause",
    "therefore": "cause",
    "so": "cause",
    "if": "implies",
    "although": "contrast",
    "but": "contrast",
}


class ProseTier1Encoder:
    adapter_id = "prose_tier1"
    version = "0.1.0"

    def encode(self, artifact: str, **kwargs) -> EncodeResult:
        clauses = [c.strip() for c in re.split(r"[.;!?]\s*", artifact) if c.strip()]
        statements: list[Statement] = []
        clause_stmts: list[Statement] = []
        for i, clause in enumerate(clauses):
            pred = first_verbish(clause) or "mentions"
            digest = blake2s(clause.encode("utf-8"), digest_size=8).hexdigest()
            clause_stmt = stmt(pred, entity(f"clause_{i}", "clause"), entity(digest, "text_digest"))
            clause_stmts.append(clause_stmt)
            statements.append(clause_stmt)
        lower = artifact.lower()
        for token, rel in CONNECTIVES.items():
            if token in lower and len(clause_stmts) >= 2:
                statements.append(stmt(rel, clause_stmts[0], clause_stmts[1]))
        return EncodeResult(
            make_case(statements or [stmt("emptyProse", entity("doc"))], {"adapter": self.adapter_id, "tier": 1}),
            ("Tier-1 prose extraction is flagged and excluded from headline claims.",),
        )


def first_verbish(text: str) -> str | None:
    words = re.findall(r"[A-Za-z]+", text)
    for word in words:
        low = word.lower()
        if low.endswith("ed") or low.endswith("ing") or low in {"is", "are", "was", "were", "has", "have"}:
            return low
    return words[0].lower() if words else None
