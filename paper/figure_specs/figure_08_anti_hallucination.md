# Figure 8 — Verifiable inference vs generative recall (H3)

**Role:** the trust figure. SMA's candidate inferences are checkable objects with
100% provenance; a context-only LLM invents entities with ~0% provenance. This is
the enterprise-credibility panel.

**Toolchain:** Matplotlib stacked/diverging bars + glyph annotations; a small
"receipt card" rendered in TikZ.

## Panel A — judged honesty (stacked bars)
For each system (SMA-grounded answers, DeepSeek context-only, local-LLM
context-only): stacked bar of judged answer cells = {correct & cited, correct
but uncited, abstained appropriately, **invented entity (confabulation)**}. The
red confabulation segment is near-zero for SMA, large for context-only. Uses the
H3 judged data (DeepSeek 99% correct / 0 invented vs local 2% / 18% confab).

## Panel B — provenance coverage (the binary contrast)
A simple, brutal 100% vs ~0% pair: fraction of surfaced claims carrying a
provenance record. SMA = 100% by construction (policy-enforced); generative
baselines ≈ 0%. Annotate "enforced in code, not prompt (sma/agent/policies.py)."

## Panel C — a real receipt (TikZ card)
One actual candidate-inference record rendered as a card: the projected
statement, `base_case_id`, `gmap_id`, `SESₙ`, the support correspondences,
skolems, `status: hypothetical`. Shows reviewers the *object*, not a claim about
it. Echoes MAMMAL's concrete structural examples.

**Data sources:** `reports/h3_judged.csv` (already produced) for A/B; a real
record from `candidate_inferences()` for C.
**Caption point:** "Every claim SMA surfaces traces to a stored case and specific
correspondences (B,C); grounding eliminates the entity confabulation that
context-only prompting exhibits (A)."
