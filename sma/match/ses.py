"""Structural evaluation score with trickle-down support."""

from __future__ import annotations

from sma.ir.schema import Statement

from .types import GMap, MatchHypothesis, node_key


def structural_evaluation(hypotheses: tuple[MatchHypothesis, ...], gamma: float = 0.25) -> float:
    by_key = {mh.key: mh for mh in hypotheses}
    parents: dict[tuple[str, str], list[tuple[str, str]]] = {mh.key: [] for mh in hypotheses}
    for mh in hypotheses:
        if not isinstance(mh.base, Statement) or not isinstance(mh.target, Statement):
            continue
        for b_arg, t_arg in zip(mh.base.args, mh.target.args, strict=True):
            child_key = (node_key(b_arg), node_key(t_arg))
            if child_key in parents:
                parents[child_key].append(mh.key)

    memo: dict[tuple[str, str], float] = {}

    def score(key: tuple[str, str], stack: frozenset[tuple[str, str]] = frozenset()) -> float:
        if key in memo:
            return memo[key]
        if key in stack:
            return by_key[key].ascension
        parent_score = sum(score(parent, stack | {key}) for parent in parents.get(key, ()))
        value = by_key[key].ascension + gamma * parent_score
        memo[key] = value
        return value

    return sum(score(key) for key in by_key)


def self_score(case, gamma: float = 0.25) -> float:
    hyps: list[MatchHypothesis] = []
    for expr in case.expressions():
        hyps.append(MatchHypothesis(expr, expr))
        for entity in expr.entities():
            hyps.append(MatchHypothesis(entity, entity))
    unique = {mh.key: mh for mh in hyps}
    return structural_evaluation(tuple(unique.values()), gamma=gamma)


def normalize_score(score: float, base, target, gamma: float = 0.25) -> float:
    denom = max(self_score(base, gamma), self_score(target, gamma), 1e-9)
    return score / denom

