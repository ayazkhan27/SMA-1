"""Dense retrieval with BAAI/bge-base-en-v1.5 (blueprint B2's specified embedder).

CPU-only. Index embeddings are batch-encoded once; queries are encoded per
call so per-query latency includes the real encode cost (same convention as
the MiniLM dense baseline in sma.eval.loghub_eval).
"""

from __future__ import annotations

import numpy as np

MODEL_NAME = "BAAI/bge-base-en-v1.5"
# Per the BGE model card, short queries in retrieval tasks should carry this
# instruction prefix; documents are encoded without it.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class BGEDenseRetriever:
    def __init__(self, model_name: str = MODEL_NAME, batch_size: int = 16):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")
        self.batch_size = batch_size
        self.doc_ids: list[str] = []
        self.doc_matrix: np.ndarray | None = None

    def build(self, documents: list[tuple[str, str]]) -> None:
        self.doc_ids = [doc_id for doc_id, _ in documents]
        texts = [text for _, text in documents]
        self.doc_matrix = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def encode_query(self, query_text: str) -> np.ndarray:
        return self.model.encode(
            [QUERY_PREFIX + query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

    def retrieve(self, query_text: str, k: int = 10) -> list[tuple[str, float]]:
        if self.doc_matrix is None:
            return []
        q = self.encode_query(query_text)
        scores = self.doc_matrix @ q  # cosine: both sides L2-normalized
        ranked = sorted(
            zip(self.doc_ids, map(float, scores)), key=lambda row: (-row[1], row[0])
        )
        return ranked[:k]
