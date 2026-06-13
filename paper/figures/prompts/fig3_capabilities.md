# Figure 3 — capabilities in action

> Assumes the **SHARED BRIEF** you already have (science context + design directives + style/palette). This file is ONLY this figure's content — do not re-derive the shared rules.

## What this panel must convey
This figure makes the abstract capabilities **concrete** with three side-by-side
mini-scenarios, each comparing **SMA (left) vs vector RAG (right)** on the *same*
input. The reader should leave thinking: "SMA is correct *and* explains itself *and*
knows its limits; RAG is confident, opaque, and can't tell when it's wrong or when
something is new." These are illustrative vignettes, not charts.

---

# FIGURE 3 — The three capabilities, in action · artboard 1500 × 900

Title top-left: **"Capabilities vector RAG cannot provide".** Three stacked rows
(a, b, c), each a left/right comparison split down the middle by a thin grey divider;
left half tinted very light teal (SMA), right half very light grey (RAG). Label the
left column "SMA" (teal) and right "Vector RAG" (grey) once at the top.

**Row a — Provenance / citation.** Input pill (centered, above the row): a rare case
"patient: 3 uncommon phenotypes". LEFT (SMA): a small structural-match glyph (two
tiny aligned graphs with dashed correspondence lines) → green chip
"✓ Diagnosis: <rare disease> — cited" with a tiny "receipt" icon listing the matched
features. RIGHT (RAG): a single dot pulled from a dotted cloud → grey chip
"Answer: <common disease>" with a red ✗ and label "similarity 0.71 — no reason given".

**Row b — Calibrated abstention.** Input pill: "case outside the knowledge base".
LEFT (SMA): the structural matcher finds nothing (an empty target graph, a small
"no match" mark) → amber chip "Abstains: no structural match". RIGHT (RAG): still
returns a nearest dot → grey chip "Confident answer (wrong)", red ✗, note "cosine
always finds a neighbour".

**Row c — Novelty detection.** Input pill: "a never-seen pattern (e.g. novel attack
chain / new disease)". LEFT (SMA): an expectation-violation burst icon → amber chip
with a flag "⚑ Flagged as novel — escalate". RIGHT (RAG): a dot snapping to the
nearest cluster → grey chip "Mislabeled as nearest known class", red ✗.

Keep each vignette compact and schematic; consistent iconography across rows
(structural-graph glyph for SMA, single-dot-in-cloud for RAG). This is the
"verifiable specialist vs confident black box" figure.

## Figure-specific sharpenings (apply with the directives)
**Single takeaway:** *SMA can cite, abstain, and flag-novelty — three things vector
RAG architecturally cannot — demonstrated on the same inputs.*
- **Row a (cite):** SMA shows a **named provenance receipt** — "✓ matched Case #214
  (rare disease X); justified by Seizure (HP:0001250), Ataxia (HP:0001251)". RAG
  returns "<common disease> ✗" with only "similarity 0.71 — no justification".
- **Row b (abstain):** SMA = a gauge with a **threshold tick τ, needle below it**,
  and "best alignment 0.31 < τ → **abstain (no answer)**". RAG = a confident **wrong**
  answer (red ✗), note "cosine always returns a neighbour".
- **Row c (novelty):** SMA = the query **fails to align with anything** in the store
  ("∅ no aligned case") → **starburst "novel — escalate"**. RAG = snaps to the nearest
  known class → "mislabeled as nearest known ✗".
Use the SAME cite/abstain/novelty icons and the SAME structured-graph + dot-cloud
glyphs as Figure A (reusable system). Pair every red with a ✗ shape.
