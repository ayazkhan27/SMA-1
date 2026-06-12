"""Per-query structural coverage indicator (blueprint section 12-R3).

The cross-system event ontology in ``logs_drain.EVENT_CLASS_RULES`` is frozen
(tag ontology-v1). When a query's vocabulary falls outside those keyword
rules, MAC/FAC retrieval degrades silently: events still get template
functors, but none of the shared cross-system class statements fire, so
structural similarity is computed over near-disjoint functor sets. This is
the "lattice-miss" tripwire from blueprint section 4.3 / 12-R3, surfaced as a
measured per-query coverage number rather than hidden.

Read-only consumer of the frozen ontology: imports ``event_classes`` from
``logs_drain`` and never modifies the rules.
"""

from __future__ import annotations

from sma.encoders.logs_drain import event_classes

# Below this fraction the SMA evidence panel shows an amber chip and the
# verbalizer prompt carries an explicit low-confidence caveat.
COVERAGE_WARN_THRESHOLD = 0.4

# Keyword attributes emitted by LogEncoder in addition to the class rules
# (timeout/retry/failure statements). They are subsets of the class keyword
# sets today, but checked explicitly so coverage stays correct even if the
# attribute rules and class rules ever diverge in a future ontology version.
_KEYWORD_ATTRIBUTES: tuple[tuple[str, ...], ...] = (
    ("timeout",),
    ("retry",),
    ("error", "exception", "fail"),
)


def _line_fired(line_lower: str, extra_classes=None) -> bool:
    if event_classes(line_lower):
        return True
    if extra_classes is not None and any(
        any(k in line_lower for k in keywords) for _, keywords in extra_classes
    ):
        return True
    return any(any(k in line_lower for k in keywords) for keywords in _KEYWORD_ATTRIBUTES)


def rule_coverage(text: str, extra_classes=None) -> dict:
    """Fraction of non-empty lines that fired at least one class rule.

    ``extra_classes``: optional (name, keywords) pairs from an active draft
    adapter - while a draft is applied, its rules legitimately count toward
    coverage (the chip must reflect what retrieval can actually use).

    Returns a dict with:
    - ``fraction``: covered / total non-empty lines (0.0 for empty text)
    - ``covered_lines`` / ``total_lines``: the raw counts
    - ``low``: True when fraction < COVERAGE_WARN_THRESHOLD
    - ``percent``: integer percent, for display
    """
    lines = [line for line in text.splitlines() if line.strip()]
    covered = sum(1 for line in lines if _line_fired(line.lower(), extra_classes))
    total = len(lines)
    fraction = covered / total if total else 0.0
    return {
        "fraction": fraction,
        "covered_lines": covered,
        "total_lines": total,
        "low": fraction < COVERAGE_WARN_THRESHOLD,
        "percent": round(fraction * 100),
    }


def coverage_warning(coverage: dict) -> str | None:
    """The exact low-confidence caveat for evidence metadata and the prompt."""
    if not coverage.get("low"):
        return None
    return (
        f"structural coverage of this query is LOW ({coverage['percent']}%) - "
        "structural retrieval is low-confidence for this vocabulary"
    )


__all__ = ["COVERAGE_WARN_THRESHOLD", "rule_coverage", "coverage_warning"]
