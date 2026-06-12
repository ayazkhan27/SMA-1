"""Structural consistency checks and kernel conflict graph."""

from __future__ import annotations

from itertools import combinations

from .types import Kernel, MatchHypothesis


def structurally_consistent(hypotheses: tuple[MatchHypothesis, ...]) -> bool:
    base_to_target: dict[str, str] = {}
    target_to_base: dict[str, str] = {}
    for mh in hypotheses:
        if mh.base_key in base_to_target and base_to_target[mh.base_key] != mh.target_key:
            return False
        if mh.target_key in target_to_base and target_to_base[mh.target_key] != mh.base_key:
            return False
        base_to_target[mh.base_key] = mh.target_key
        target_to_base[mh.target_key] = mh.base_key
    return True


def kernels_conflict(left: Kernel, right: Kernel) -> bool:
    # Iterate over the smaller binding table; both tables are cached on the
    # kernels, so each check is O(min(|left|, |right|)) hash probes.
    if len(left.bindings) > len(right.bindings):
        left, right = right, left
    right_bindings = right.bindings
    right_reverse = right.reverse_bindings
    for b_key, t_key in left.bindings.items():
        other_t = right_bindings.get(b_key)
        if other_t is not None and other_t != t_key:
            return True
        other_b = right_reverse.get(t_key)
        if other_b is not None and other_b != b_key:
            return True
    return False


def conflict_edges(kernels: tuple[Kernel, ...]) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for i, j in combinations(range(len(kernels)), 2):
        if kernels_conflict(kernels[i], kernels[j]):
            edges.add((i, j))
    return edges

