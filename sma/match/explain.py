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

