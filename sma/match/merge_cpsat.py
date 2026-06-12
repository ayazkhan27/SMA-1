"""Exact-anytime MWIS merge using Google OR-Tools CP-SAT solver."""

from __future__ import annotations

from .conflicts import conflict_edges
from .merge_greedy import greedy_merge
from .types import Kernel

try:
    from ortools.sat.python import cp_model
except ImportError:  # pragma: no cover - ortools is a hard dependency in pyproject
    cp_model = None

# Brute-force enumeration cap when ortools is unavailable (2^12 masks max).
_BRUTE_FORCE_LIMIT = 12


def exact_or_greedy_merge(
    kernels: tuple[Kernel, ...],
    exact_limit: int = 60,
    time_budget_ms: int = 20,
) -> tuple[tuple[Kernel, ...], float | None]:
    """Select a maximum-weight independent set of kernels.

    Returns (selected kernels, optimality gap). Gap 0.0 means certified
    optimal; None means the greedy/anytime fallback was used without a
    certificate.
    """
    if not kernels:
        return (), 0.0

    # Greedy is the published O(n^2 log n) fallback; use it directly when the
    # MIP model would cost more than it buys (large conflict-kernel counts).
    if len(kernels) > exact_limit:
        return greedy_merge(kernels), None

    edges = conflict_edges(kernels)

    if cp_model is not None:
        model = cp_model.CpModel()
        n = len(kernels)
        x = [model.NewBoolVar(f"x_{i}") for i in range(n)]
        for i, j in edges:
            model.Add(x[i] + x[j] <= 1)
        # CP-SAT needs integer weights; 1e5 scaling keeps 5 decimal places.
        scaled_weights = [int(round(max(k.weight, 0.0) * 100000)) for k in kernels]
        model.Maximize(sum(scaled_weights[i] * x[i] for i in range(n)))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_budget_ms / 1000.0
        status = solver.Solve(model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            selected = tuple(kernels[i] for i in range(n) if solver.Value(x[i]))
            gap = 0.0 if status == cp_model.OPTIMAL else None
            return selected, gap
        return greedy_merge(kernels), None

    # No ortools: exact enumeration for tiny instances, greedy otherwise.
    n = len(kernels)
    if n > _BRUTE_FORCE_LIMIT:
        return greedy_merge(kernels), None
    best_mask = 0
    best_weight = float("-inf")
    for mask in range(1 << n):
        ok = True
        for i, j in edges:
            if (mask & (1 << i)) and (mask & (1 << j)):
                ok = False
                break
        if not ok:
            continue
        weight = sum(kernels[i].weight for i in range(n) if mask & (1 << i))
        if weight > best_weight:
            best_weight = weight
            best_mask = mask
    selected = tuple(kernels[i] for i in range(n) if best_mask & (1 << i))
    return selected, 0.0
