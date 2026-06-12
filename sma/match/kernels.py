"""Kernel construction."""

from __future__ import annotations

from sma.ir.canon import Canonicalizer, default_canonicalizer
from sma.ir.schema import Case

from sma.ir.schema import Statement

from .conflicts import structurally_consistent
from .mh import seed_expression_mhs, support_closure
from .ses import structural_evaluation
from .types import Kernel, MatchConfig, node_key


def build_kernels(
    base: Case,
    target: Case,
    config: MatchConfig | None = None,
    canon: Canonicalizer | None = None,
    cost_fn=None,
) -> tuple[Kernel, ...]:
    config = config or MatchConfig()
    canon = canon or default_canonicalizer()
    seeds = seed_expression_mhs(base.expressions(), target.expressions(), config=config, canon=canon)
    # Root MHs only (blueprint section 2.2): a seed that appears as an argument
    # pair of another seed is covered by its parent's support closure, so giving
    # it its own kernel only inflates the merge problem.
    child_keys: set[tuple[str, str]] = set()
    for seed in seeds:
        if isinstance(seed.base, Statement) and isinstance(seed.target, Statement):
            for b_arg, t_arg in zip(seed.base.args, seed.target.args):
                if isinstance(b_arg, Statement) and isinstance(t_arg, Statement):
                    child_keys.add((node_key(b_arg), node_key(t_arg)))
    kernels: list[Kernel] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    for seed in seeds:
        if seed.key in child_keys:
            continue
        closure = support_closure(seed, canon=canon, delta=config.delta, rho=config.rho)
        if closure is None:
            # Root pairs unequal constants (e.g. different template-name
            # entities under count): structurally impossible, discard.
            continue
        if not structurally_consistent(closure):
            continue
        key = tuple(sorted(mh.key for mh in closure))
        if key in seen:
            continue
        seen.add(key)
        weight = structural_evaluation(closure, gamma=config.gamma, cost_fn=cost_fn)
        kernels.append(Kernel(seed, closure, weight=weight))
    return tuple(kernels)

