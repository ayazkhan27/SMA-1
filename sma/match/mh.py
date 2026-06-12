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


# Entity types whose names are CONSTANTS, not variables (blueprint 2.1:
# entities/constants are distinct vocabulary classes). A template-name or
# integer entity denotes itself; pairing count(template_A, 3) with
# count(template_B, 2) is vacuous shape-matching, not analogy - it was the
# root cause of the Liberty haystack failure (generic bookkeeping skeleton
# matching any session against any other).
# Integers are deliberately NOT constants: count(template_X, 3) vs
# count(template_X, 5) is a legitimate analogy (same burst, different size);
# the template-name constraint alone blocks the vacuous cross-template case.
CONSTANT_ENTITY_TYPES = frozenset({"event_type"})


def constants_compatible(b_ent: Entity, t_ent: Entity) -> bool:
    if b_ent.type in CONSTANT_ENTITY_TYPES and t_ent.type in CONSTANT_ENTITY_TYPES:
        return b_ent.name == t_ent.name
    return True


def support_closure(
    root: MatchHypothesis,
    canon: Canonicalizer | None = None,
    delta: int = 0,
    rho: float = 1.0,
) -> tuple[MatchHypothesis, ...] | None:
    """Downward closure of a root MH; None when structurally impossible.

    SME parallel connectivity: argument correspondences must themselves be
    LEGAL match hypotheses. A statement-argument pair with incompatible
    functors invalidates the whole kernel (previously it was silently
    admitted, letting higher-order parents like `before` manufacture
    cross-template "matches" that surprisal weighting then amplified - the
    Liberty ses_n>1 anomaly). Compatibility = canonical identity, or lattice
    ascension within delta at rho^dist penalty. Unequal constants likewise
    invalidate.
    """
    canon = canon or default_canonicalizer()
    out: list[MatchHypothesis] = []
    seen: set[tuple[str, str]] = set()
    bad = False

    def add(mh: MatchHypothesis) -> None:
        nonlocal bad
        if bad or mh.key in seen:
            return
        seen.add(mh.key)
        out.append(mh)
        if isinstance(mh.base, Statement) and isinstance(mh.target, Statement):
            if mh.base.arity != mh.target.arity:
                bad = True
                return
            for b_arg, t_arg in zip(mh.base.args, mh.target.args, strict=True):
                if isinstance(b_arg, Entity) and isinstance(t_arg, Entity):
                    if not constants_compatible(b_arg, t_arg):
                        bad = True
                        return
                    add(MatchHypothesis(b_arg, t_arg))
                elif isinstance(b_arg, Statement) and isinstance(t_arg, Statement):
                    ok, asc, ancestor, dist = canon.compatible(
                        b_arg.functor, t_arg.functor, delta=delta, rho=rho
                    )
                    if not ok:
                        bad = True
                        return
                    add(MatchHypothesis(b_arg, t_arg, ascension=mh.ascension * asc,
                                        ancestor=ancestor, distance=dist))
                else:
                    # Statement paired with entity (or vice versa): illegal.
                    bad = True
                    return

    add(root)
    return None if bad else tuple(out)
