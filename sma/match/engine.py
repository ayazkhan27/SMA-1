"""Top-level mapping engine."""

from __future__ import annotations

from sma.ir.canon import Canonicalizer, default_canonicalizer
from sma.ir.schema import Case

from .kernels import build_kernels
from .mdl import mdl_gain
from .merge_cpsat import exact_or_greedy_merge
from .ses import normalize_score, structural_evaluation
from .types import GMap, MatchConfig, MatchHypothesis


def match_cases(
    base: Case,
    target: Case,
    config: MatchConfig | None = None,
    canon: Canonicalizer | None = None,
) -> GMap:
    config = config or MatchConfig()
    canon = canon or default_canonicalizer()
    kernels = build_kernels(base, target, config=config, canon=canon)
    selected, gap = exact_or_greedy_merge(
        kernels, exact_limit=config.exact_kernel_limit, time_budget_ms=config.cpsat_time_ms
    )
    unique: dict[tuple[str, str], MatchHypothesis] = {}
    for kernel in selected:
        for mh in kernel.hypotheses:
            unique[mh.key] = mh
    hypotheses = tuple(unique.values())
    if config.scorer == "mdl":
        score = mdl_gain(hypotheses, target)
        normalized = score / max(len(target.expressions()), 1)
    else:
        score = structural_evaluation(hypotheses, gamma=config.gamma)
        normalized = normalize_score(score, base, target, gamma=config.gamma)
    return GMap(
        base=base,
        target=target,
        hypotheses=hypotheses,
        kernels=selected,
        score=score,
        normalized_score=normalized,
        scorer=config.scorer,
        optimality_gap=gap,
    )

