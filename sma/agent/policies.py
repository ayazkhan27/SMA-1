"""Hard API policies for SMA agent memory."""

from __future__ import annotations


def reject_free_text_facts(annotation) -> None:
    if isinstance(annotation, str):
        raise ValueError("free-text facts are rejected; route annotations through encode()")
    if isinstance(annotation, dict) and "sexpr" not in annotation and "case_id" not in annotation:
        raise ValueError("annotations must be encoded cases or canonical S-expressions")


def require_provenance(claims: list[dict]) -> list[dict]:
    checked: list[dict] = []
    for claim in claims:
        if not claim.get("provenance"):
            claim = dict(claim)
            claim["status"] = "unsupported-by-memory"
        checked.append(claim)
    return checked

