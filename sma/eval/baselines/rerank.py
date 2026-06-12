"""Cross-encoder reranking (cross-encoder/ms-marco-MiniLM-L-6-v2).

Reranks a candidate pool (here: top-20 of Hybrid-RRF) and returns the top-k.
Raw model outputs are logits; we pass them through a sigmoid so the scores
are positive and usable by the protocol's weighted top-5 label vote (a
negative weight would flip a vote, which is not what 'weighted' means there).
Candidate texts are truncated to ``max_chars`` before pairing; the tokenizer
truncates to 512 wordpieces anyway, so this only bounds tokenizer cost.
"""

from __future__ import annotations

import numpy as np

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    def __init__(self, model_name: str = MODEL_NAME, max_chars: int = 2000):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device="cpu", max_length=512)
        self.max_chars = max_chars

    def rerank(
        self,
        query_text: str,
        candidates: list[tuple[str, str]],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        if not candidates:
            return []
        q = query_text[: self.max_chars]
        pairs = [(q, text[: self.max_chars]) for _, text in candidates]
        logits = self.model.predict(pairs, show_progress_bar=False)
        scores = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
        ranked = sorted(
            zip((doc_id for doc_id, _ in candidates), map(float, scores)),
            key=lambda row: (-row[1], row[0]),
        )
        return ranked[:top_k]
