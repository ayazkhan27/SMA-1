"""SPLADE learned sparse retrieval (naver/splade-cocondenser-ensembledistil).

CPU-only. Standard SPLADE document/query representation:
    rep = max over token positions of log(1 + relu(MLM logits)),
masked by attention. Documents are batch-encoded once into a scipy CSR
matrix; query scoring is a sparse dot product. Per-query latency includes
the query forward pass (same convention as the other neural baselines).

Inputs are truncated to ``max_length`` wordpieces (default 256) to keep the
CPU forward pass tractable on long log sessions; the same truncation applies
to documents and queries, so no method gets privileged context.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

MODEL_NAME = "naver/splade-cocondenser-ensembledistil"


class SpladeRetriever:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        max_length: int = 256,
        batch_size: int = 8,
    ):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.eval()
        self.max_length = max_length
        self.batch_size = batch_size
        self.doc_ids: list[str] = []
        self.doc_matrix: sparse.csr_matrix | None = None

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        torch = self.torch
        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        with torch.no_grad():
            logits = self.model(**tokens).logits  # (B, T, V)
        # log(1 + relu(logits)), max-pooled over valid token positions.
        weights = torch.log1p(torch.relu(logits))
        mask = tokens["attention_mask"].unsqueeze(-1)
        reps = (weights * mask).max(dim=1).values  # (B, V)
        return reps.numpy()

    def encode(self, texts: list[str]) -> sparse.csr_matrix:
        rows = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            rows.append(sparse.csr_matrix(self._encode_batch(batch)))
        return sparse.vstack(rows) if rows else sparse.csr_matrix((0, 0))

    def build(self, documents: list[tuple[str, str]]) -> None:
        self.doc_ids = [doc_id for doc_id, _ in documents]
        self.doc_matrix = self.encode([text for _, text in documents])

    def retrieve(self, query_text: str, k: int = 10) -> list[tuple[str, float]]:
        if self.doc_matrix is None:
            return []
        q = sparse.csr_matrix(self._encode_batch([query_text]))
        scores = np.asarray((self.doc_matrix @ q.T).todense()).ravel()
        ranked = sorted(
            zip(self.doc_ids, map(float, scores)), key=lambda row: (-row[1], row[0])
        )
        return ranked[:k]
