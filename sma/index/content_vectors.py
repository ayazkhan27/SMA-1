"""MAC content vectors with WL-1 refinement features."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from sma.ir.canon import Canonicalizer, default_canonicalizer
from sma.ir.schema import Case, Statement


Vector = Counter[str]


def functor_vector(
    case: Case, wl: bool = True, canon: Canonicalizer | None = None, canonicalize: bool = True
) -> Vector:
    canon = canon or default_canonicalizer()
    counts: Vector = Counter()
    for expr in case.expressions():
        functor = canon.canonical(expr.functor) if canonicalize else expr.functor
        counts[f"f:{functor}"] += 1
        if wl:
            for i, arg in enumerate(expr.args):
                if isinstance(arg, Statement):
                    child_functor = canon.canonical(arg.functor) if canonicalize else arg.functor
                    counts[f"wl:{functor}:{i}:{child_functor}"] += 1
                else:
                    counts[f"wl:{functor}:{i}:ENTITY"] += 1
    return counts


def cosine(left: Vector, right: Vector) -> float:
    dot = sum(v * right.get(k, 0) for k, v in left.items())
    left_norm = sum(v * v for v in left.values()) ** 0.5
    right_norm = sum(v * v for v in right.values()) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


@dataclass(frozen=True)
class CaseVector:
    case_id: str
    vector: Vector
