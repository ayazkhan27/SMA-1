"""Pre-registered statistics for the confirmatory battery (prereg section 5).

Per-query paired bootstrap (10,000 resamples) for SMA-vs-baseline deltas
with 95% percentile CIs, Holm-Bonferroni step-down correction within each
dataset's family of baseline comparisons, and Cliff's delta as the effect
size. Everything is deterministic: the bootstrap uses an explicitly seeded
numpy Generator and no global RNG state.
"""

from __future__ import annotations

import numpy as np

DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 12345

# Resample index matrices are drawn in chunks so a 10k-resample bootstrap
# over thousands of pooled queries stays within a few tens of MB.
_CHUNK = 1_000


def paired_bootstrap(
    a: list[float],
    b: list[float],
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Paired bootstrap of mean(a - b) over per-query scores.

    ``a`` and ``b`` are per-query scores for two methods on the SAME queries
    (paired by position). Returns::

        {"delta":   observed mean(a - b),
         "ci_low":  2.5th percentile of the bootstrap distribution,
         "ci_high": 97.5th percentile,
         "p_value": two-sided bootstrap p for delta != 0}

    The p-value is the doubled smaller tail of the bootstrap distribution
    around zero, with a +1/(R+1) correction so it is never exactly 0.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.ndim != 1 or b_arr.ndim != 1:
        raise ValueError("paired_bootstrap expects 1-D score lists")
    if a_arr.shape != b_arr.shape:
        raise ValueError(
            f"paired scores must have equal length (got {a_arr.size} vs {b_arr.size})"
        )
    if a_arr.size == 0:
        raise ValueError("paired_bootstrap requires at least one paired observation")
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")

    diffs = a_arr - b_arr
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_resamples, dtype=float)
    done = 0
    while done < n_resamples:
        size = min(_CHUNK, n_resamples - done)
        idx = rng.integers(0, diffs.size, size=(size, diffs.size))
        deltas[done : done + size] = diffs[idx].mean(axis=1)
        done += size

    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    p_low = (np.count_nonzero(deltas <= 0.0) + 1) / (n_resamples + 1)
    p_high = (np.count_nonzero(deltas >= 0.0) + 1) / (n_resamples + 1)
    return {
        "delta": float(diffs.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(min(1.0, 2.0 * min(p_low, p_high))),
    }


def holm_bonferroni(p_values: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values, keyed like the input.

    Sort the m raw p-values ascending; the i-th (1-based) is multiplied by
    (m - i + 1), running maxima enforce monotonicity, and everything is
    capped at 1.0. Ties are processed in sorted (p, key) order, which does
    not affect the adjusted values.
    """
    m = len(p_values)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (key, p) in enumerate(sorted(p_values.items(), key=lambda kv: (kv[1], kv[0]))):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p-value for {key!r} outside [0, 1]: {p}")
        running = max(running, (m - i) * p)
        adjusted[key] = min(1.0, running)
    return adjusted


def cliffs_delta(a: list[float], b: list[float]) -> float:
    """Standard Cliff's delta in [-1, 1]: P(a > b) - P(a < b) over all pairs.

    +1 means every a exceeds every b; -1 the reverse; 0 means stochastic
    equality. Computed via sorted ranks (O((n+m) log m)), so pooled
    multi-seed score lists are fine.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.size == 0 or b_arr.size == 0:
        raise ValueError("cliffs_delta requires non-empty score lists")
    b_sorted = np.sort(b_arr)
    n_b_below = np.searchsorted(b_sorted, a_arr, side="left").sum()
    n_b_above = (b_arr.size - np.searchsorted(b_sorted, a_arr, side="right")).sum()
    return float((int(n_b_below) - int(n_b_above)) / (a_arr.size * b_arr.size))
