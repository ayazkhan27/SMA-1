"""Parameter-free MDL-like scorer."""

from __future__ import annotations

import math
from collections import Counter

from sma.ir.schema import Case

from .types import MatchHypothesis


def corpus_functor_costs(cases: list[Case]) -> dict[str, float]:
    counts: Counter[str] = Counter()
    total = 0
    for case in cases:
        for functor, n in case.functor_counts().items():
            counts[functor] += n
            total += n
    vocab = max(len(counts), 1)
    return {functor: -math.log2((count + 0.5) / (total + 0.5 * vocab)) for functor, count in counts.items()}


def mdl_gain(hypotheses: tuple[MatchHypothesis, ...], target: Case) -> float:
    costs = corpus_functor_costs([target])
    matched_target_exprs = {
        mh.target.functor for mh in hypotheses if hasattr(mh.target, "functor")
    }
    return sum(costs.get(functor, 1.0) for functor in matched_target_exprs)

