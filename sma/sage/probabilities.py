"""Frequency probabilities for SAGE facts."""

from __future__ import annotations


def support_probability(count: int, total: int, alpha: float = 1.0) -> float:
    if total <= 0:
        return 0.0
    return (count + alpha) / (total + 2 * alpha)

