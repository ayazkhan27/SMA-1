"""MAC content vectors with WL-1 refinement features."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from sma.ir.canon import Canonicalizer, default_canonicalizer
from sma.ir.schema import Case, Statement


Vector = Counter[str]


def functor_vector(
    case: Case,
    wl: bool = True,
    canon: Canonicalizer | None = None,
    canonicalize: bool = True,
    delta: int = 0,
) -> Vector:
    """MAC content vector: canonical functor counts + WL-1 features.

    With delta > 0, each functor also contributes its lattice ancestors within
    delta steps (blueprint 2.7: counts over the <=delta ancestor closure), so
    vocabularies bridged only by the lattice still intersect at the MAC stage.
    Ancestor features only ADD mass, keeping the Lemma-2 bound admissible.
    """
    canon = canon or default_canonicalizer()
    counts: Vector = Counter()
    for expr in case.expressions():
        functor = canon.canonical(expr.functor) if canonicalize else expr.functor
        counts[f"f:{functor}"] += 1
        if delta:
            for ancestor, dist in canon.lattice.ancestors(functor, delta).items():
                if ancestor != functor:
                    counts[f"f:{ancestor}"] += 1
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
