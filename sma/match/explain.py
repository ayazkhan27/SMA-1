"""Human-readable mapping explanations."""

from __future__ import annotations

from .types import GMap


def correspondence_table(gmap: GMap) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for mh in gmap.hypotheses:
        rows.append(
            {
                "base": mh.base_key,
                "target": mh.target_key,
                "ascension": f"{mh.ascension:.3f}",
                "ancestor": mh.ancestor or "",
            }
        )
    return rows


def explain_text(gmap: GMap) -> str:
    lines = [
        f"gmap score={gmap.score:.3f} SES_n={gmap.normalized_score:.3f} scorer={gmap.scorer}",
        "correspondences:",
    ]
    for row in correspondence_table(gmap):
        lines.append(f"- {row['base']} -> {row['target']} asc={row['ascension']} {row['ancestor']}")
    return "\n".join(lines)



def alignment_summary(gmap: GMap) -> str:
    """One-line 'why this precedent matched' for verbalizers and evidence panels.

    Aggregates matched statement-level correspondences by canonical functor so
    an LLM (or human) sees the shared STRUCTURE (e.g. 'kernelEvent x4,
    failureEvent x4, before x3') instead of re-deriving similarity from raw
    text semantics.
    """
    from collections import Counter

    from sma.ir.canon import default_canonicalizer
    from sma.ir.schema import Statement

    canon = default_canonicalizer()
    matched: Counter[str] = Counter()
    for mh in gmap.hypotheses:
        if isinstance(mh.base, Statement):
            functor = canon.canonical(mh.base.functor)
            if functor != "logSession":
                matched[functor] += 1
    if not matched:
        return "no statement-level correspondences"
    parts = ", ".join(
        f"{functor} x{count}" if count > 1 else functor
        for functor, count in matched.most_common(6)
    )
    return f"shared structure: {parts}; ses_n={gmap.normalized_score:.2f}"
