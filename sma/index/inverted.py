"""Exact histogram-intersection upper bounds."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .content_vectors import Vector


def histogram_intersection(left: Vector, right: Vector) -> int:
    if len(left) > len(right):
        left, right = right, left
    return sum(min(value, right.get(key, 0)) for key, value in left.items())


def ses_upper_bound(left: Vector, right: Vector, max_score_per_mh: float = 2.0) -> float:
    return max_score_per_mh * histogram_intersection(left, right)


@dataclass
class InvertedIndex:
    postings: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    vectors: dict[str, Vector] = field(default_factory=dict)

    def add(self, case_id: str, vector: Vector) -> None:
        self.vectors[case_id] = vector
        for feature in vector:
            self.postings.setdefault(feature, set()).add(case_id)

    def candidates(self, query: Vector) -> set[str]:
        out: set[str] = set()
        for feature in query:
            out.update(self.postings.get(feature, ()))
        return out

    def bound(self, query: Vector, case_id: str, max_score_per_mh: float = 2.0) -> float:
        return ses_upper_bound(query, self.vectors[case_id], max_score_per_mh=max_score_per_mh)

