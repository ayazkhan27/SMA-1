"""Simple lexical baseline."""

from __future__ import annotations

from collections import Counter


def lexical_score(query: str, document: str) -> float:
    q = Counter(query.lower().split())
    d = Counter(document.lower().split())
    return sum(min(v, d.get(k, 0)) for k, v in q.items())


def rank_bm25_like(query: str, documents: list[tuple[str, str]], k: int = 10) -> list[tuple[str, float]]:
    rows = [(doc_id, lexical_score(query, text)) for doc_id, text in documents]
    rows.sort(key=lambda row: (-row[1], row[0]))
    return rows[:k]

