"""Metrics for the agentic ontology suite.

Three headline metrics, matching the design spec (prereg section 4):

* ``tail_topk`` — per-method top-k accuracy on ALL queries and on the
  registered RARE slice (the tail). Top-k accuracy is the fraction of
  rows whose true-entity rank is <= k, reusing the convention from
  ``sma.eval.ontology_bench``.
* ``risk_coverage_aurc`` — cite-or-abstain selective-prediction curve:
  sort by confidence (desc), sweep coverage 0->1, risk is the cumulative
  error rate over the covered head. AURC is the mean risk over the sweep;
  LOWER is better (a well-calibrated ranker keeps its mistakes for last).
* ``novelty_f1`` — F1 of predicted-novel flags vs the truly held-out set.
"""

from __future__ import annotations

ABSENT_RANK = 999


def tail_topk(
    rows: list[dict],
    k: int,
) -> dict[str, dict[str, float]]:
    """Per-method top-k accuracy on the ALL slice and the RARE slice.

    ``rows`` is a list of ``{method_name: rank, "rare": bool}`` dicts, where
    ``rank`` is the 1-based rank of the true entity (``ABSENT_RANK`` if it
    never surfaced). Returns ``{method: {"all": acc, "rare": acc}}`` where
    ``acc`` is the fraction of rows with ``rank <= k``.
    """
    methods = sorted({m for r in rows for m in r if m != "rare"})
    rare_rows = [r for r in rows if r.get("rare")]
    out: dict[str, dict[str, float]] = {}
    for m in methods:
        out[m] = {
            "all": _topk_acc(rows, m, k),
            "rare": _topk_acc(rare_rows, m, k),
        }
    return out


def _topk_acc(rows: list[dict], method: str, k: int) -> float:
    """Fraction of ``rows`` whose ``method`` rank is <= k."""
    if not rows:
        return 0.0
    hits = sum(1 for r in rows if r.get(method, ABSENT_RANK) <= k)
    return hits / len(rows)


def risk_coverage_aurc(
    confidences: list[float],
    correct: list[bool],
) -> tuple[float, list[tuple[float, float]]]:
    """Area under the risk-coverage curve (lower is better).

    Items are sorted by ``confidences`` descending; coverage sweeps from the
    most-confident prediction to all of them. At each coverage point the risk
    is the cumulative error rate over the covered head. Returns the AURC (mean
    risk over the sweep) and the ``(coverage, risk)`` curve points.
    """
    n = len(correct)
    if n == 0:
        return 0.0, []
    order = sorted(range(n), key=lambda i: -confidences[i])
    cum_err = 0
    pts: list[tuple[float, float]] = []
    for j, i in enumerate(order, 1):
        cum_err += 0 if correct[i] else 1
        pts.append((j / n, cum_err / j))  # (coverage, risk)
    aurc = sum(r for _, r in pts) / max(len(pts), 1)
    return aurc, pts


def novelty_f1(pred: list[bool], truth: list[bool]) -> float:
    """F1 of predicted-novel flags vs the truly held-out (novel) set."""
    tp = sum(1 for p, t in zip(pred, truth) if p and t)
    fp = sum(1 for p, t in zip(pred, truth) if p and not t)
    fn = sum(1 for p, t in zip(pred, truth) if not p and t)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0
