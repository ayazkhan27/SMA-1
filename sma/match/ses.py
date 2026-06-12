"""Structural evaluation score with trickle-down support.

Two weighting regimes share this code path:
- SES (default): every match hypothesis carries unit base weight sigma_0 = 1.
- surprisal-SES (score-v2 candidate, ADR-004 upgrade path): statement MHs
  carry sigma_0 = corpus surprisal of their canonical functor (-log2 p), so
  rare shared structure counts more while systematicity still compounds via
  trickle-down. With cost_fn=None this reduces exactly to SES.
"""

from __future__ import annotations

from typing import Callable

from sma.ir.schema import Statement

from .types import GMap, MatchHypothesis, node_key

CostFn = Callable[[MatchHypothesis], float]


def structural_evaluation(
    hypotheses: tuple[MatchHypothesis, ...],
    gamma: float = 0.25,
    cost_fn: CostFn | None = None,
) -> float:
    by_key = {mh.key: mh for mh in hypotheses}
    parents: dict[tuple[str, str], list[tuple[str, str]]] = {mh.key: [] for mh in hypotheses}
    for mh in hypotheses:
        if not isinstance(mh.base, Statement) or not isinstance(mh.target, Statement):
            continue
        for b_arg, t_arg in zip(mh.base.args, mh.target.args, strict=True):
            child_key = (node_key(b_arg), node_key(t_arg))
            if child_key in parents:
                parents[child_key].append(mh.key)

    def weight(mh: MatchHypothesis) -> float:
        return 1.0 if cost_fn is None else cost_fn(mh)

    memo: dict[tuple[str, str], float] = {}

    def score(key: tuple[str, str], stack: frozenset[tuple[str, str]] = frozenset()) -> float:
        if key in memo:
            return memo[key]
        if key in stack:
            return weight(by_key[key]) * by_key[key].ascension
        parent_score = sum(score(parent, stack | {key}) for parent in parents.get(key, ()))
        value = weight(by_key[key]) * by_key[key].ascension + gamma * parent_score
        memo[key] = value
        return value

    return sum(score(key) for key in by_key)


def self_score(case, gamma: float = 0.25, cost_fn: CostFn | None = None) -> float:
    hyps: list[MatchHypothesis] = []
    for expr in case.expressions():
        hyps.append(MatchHypothesis(expr, expr))
        for entity in expr.entities():
            hyps.append(MatchHypothesis(entity, entity))
    unique = {mh.key: mh for mh in hyps}
    return structural_evaluation(tuple(unique.values()), gamma=gamma, cost_fn=cost_fn)


def normalize_score(
    score: float, base, target, gamma: float = 0.25, cost_fn: CostFn | None = None
) -> float:
    # Same weights in numerator and denominators keep ses_n scale-free.
    denom = max(
        self_score(base, gamma, cost_fn=cost_fn),
        self_score(target, gamma, cost_fn=cost_fn),
        1e-9,
    )
    return score / denom
