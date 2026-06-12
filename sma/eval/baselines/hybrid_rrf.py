"""Reciprocal-rank fusion (blueprint B3: the strong practical RAG).

Fuses any number of ranked lists with the standard RRF constant k=60
(Cormack, Clarke & Buettcher 2009). Scores are sum(1 / (k + rank)) with
1-based ranks; ties break on doc id for determinism.
"""

from __future__ import annotations

RRF_K = 60


def rrf_fuse(
    rankings: list[list[tuple[str, float]]],
    k: int = RRF_K,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked (doc_id, score) lists by reciprocal rank.

    Input scores are ignored; only rank order matters (that is the point of
    RRF -- it is scale-free across heterogeneous retrievers).
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    ranked = sorted(fused.items(), key=lambda row: (-row[1], row[0]))
    if top_k is not None:
        ranked = ranked[:top_k]
    return ranked
