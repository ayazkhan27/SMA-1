"""Match hypothesis seeding and support closure."""

from __future__ import annotations

from collections import defaultdict, deque

from sma.ir.canon import Canonicalizer, default_canonicalizer
from sma.ir.schema import Entity, Statement
from sma.ir.sexpr import dumps_node

from .types import MatchConfig, MatchHypothesis


def seed_expression_mhs(
    base_exprs: tuple[Statement, ...],
    target_exprs: tuple[Statement, ...],
    config: MatchConfig | None = None,
    canon: Canonicalizer | None = None,
) -> tuple[MatchHypothesis, ...]:
    config = config or MatchConfig()
    canon = canon or default_canonicalizer()
    base_groups = _group_by_signature(base_exprs, canon)
    target_groups = _group_by_signature(target_exprs, canon)

    out: list[MatchHypothesis] = []
    for key, bases in base_groups.items():
        targets = target_groups.get(key)
        if targets:
            for b, t in _capped_pairs(bases, targets, config.mh_group_cap):
                out.append(MatchHypothesis(b, t))
    if config.delta > 0:
        # Minimal ascension across canonical groups, also capped per group pair.
        for (b_functor, b_arity), bases in base_groups.items():
            for (t_functor, t_arity), targets in target_groups.items():
                if b_arity != t_arity or b_functor == t_functor:
                    continue
                ok, asc, ancestor, dist = canon.compatible(
                    b_functor, t_functor, delta=config.delta, rho=config.rho
                )
                if ok:
                    for b, t in _capped_pairs(bases, targets, config.mh_group_cap):
                        out.append(MatchHypothesis(b, t, asc, ancestor, dist))
    return tuple(out)


def _group_by_signature(
    exprs: tuple[Statement, ...], canon: Canonicalizer
) -> dict[tuple[str, int], list[Statement]]:
    groups: dict[tuple[str, int], list[Statement]] = defaultdict(list)
    for expr in exprs:
        groups[(canon.canonical(expr.functor), expr.arity)].append(expr)
    return groups


def _capped_pairs(
    bases: list[Statement], targets: list[Statement], cap: int
) -> list[tuple[Statement, Statement]]:
    """Deterministic U-ordered pair selection within one functor group.

    Small groups keep the full cross product. Large groups are capped:
    bit-identical statements pair first (they carry the highest achievable
    match score), then a band around the canonical sort order fills the rest.
    """
    if len(bases) * len(targets) <= cap:
        return [(b, t) for b in bases for t in targets]

    base_text = {id(b): dumps_node(b) for b in bases}
    target_text = {id(t): dumps_node(t) for t in targets}
    sorted_bases = sorted(bases, key=lambda s: base_text[id(s)])
    sorted_targets = sorted(targets, key=lambda s: target_text[id(s)])

    pairs: list[tuple[Statement, Statement]] = []
    used: set[tuple[int, int]] = set()

    by_text: dict[str, deque] = defaultdict(deque)
    for t in sorted_targets:
        by_text[target_text[id(t)]].append(t)
    for b in sorted_bases:
        queue = by_text.get(base_text[id(b)])
        if queue:
            t = queue.popleft()
            pairs.append((b, t))
            used.add((id(b), id(t)))
            if len(pairs) >= cap:
                return pairs

    n_targets = len(sorted_targets)
    for offset in range(n_targets):
        for i, b in enumerate(sorted_bases):
            for j in ((i + offset, i - offset) if offset else (i,)):
                if 0 <= j < n_targets:
                    t = sorted_targets[j]
                    if (id(b), id(t)) not in used:
                        used.add((id(b), id(t)))
                        pairs.append((b, t))
                        if len(pairs) >= cap:
                            return pairs
    return pairs


def support_closure(root: MatchHypothesis) -> tuple[MatchHypothesis, ...]:
    out: list[MatchHypothesis] = []
    seen: set[tuple[str, str]] = set()

    def add(mh: MatchHypothesis) -> None:
        if mh.key in seen:
            return
        seen.add(mh.key)
        out.append(mh)
        if isinstance(mh.base, Statement) and isinstance(mh.target, Statement):
            if mh.base.arity != mh.target.arity:
                return
            for b_arg, t_arg in zip(mh.base.args, mh.target.args, strict=True):
                if isinstance(b_arg, Entity) and isinstance(t_arg, Entity):
                    add(MatchHypothesis(b_arg, t_arg))
                elif isinstance(b_arg, Statement) and isinstance(t_arg, Statement):
                    add(MatchHypothesis(b_arg, t_arg, ascension=mh.ascension))

    add(root)
    return tuple(out)
