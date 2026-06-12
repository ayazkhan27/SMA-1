"""Small ANN facade with deterministic brute-force fallback."""

from __future__ import annotations

from dataclasses import dataclass, field

from .content_vectors import Vector, cosine


@dataclass
class AnnIndex:
    vectors: dict[str, Vector] = field(default_factory=dict)

    def add(self, case_id: str, vector: Vector) -> None:
        self.vectors[case_id] = vector

    def search(self, query: Vector, k: int = 200) -> list[tuple[str, float]]:
        ranked = [(case_id, cosine(query, vector)) for case_id, vector in self.vectors.items()]
        ranked.sort(key=lambda row: (-row[1], row[0]))
        return ranked[:k]

