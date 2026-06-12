# Figure 7 — Agent drift: re-derived memory does not compound error (H4)

**Role:** the differentiator that makes SMA *agentic*, not just retrieval. The
claim: because SMA re-derives state from the environment via the deterministic
encoder (beliefs live in the case store, not the context window), per-step error
does not compound autoregressively the way context-accumulating agents do.

**Toolchain:** Matplotlib (SciencePlots), matplotlib-label-lines for inline curve
labels (no legend clutter), a small TikZ inset for the mechanism contrast.

## Panel A — fidelity decay over horizon
x = horizon t (10…200 steps), y = state-fidelity F1 (agent beliefs vs ground
truth, probed). Three curves: **context-only** (steepest decay), **RAG-notes**
(stores its own generated text → still compounds), **SMA** (near-flat). Bands =
seed variability. Inline labels via label-lines. The shallow SMA slope is the
result.

## Panel B — decay-slope bars
OLS slope of F1 vs t for each memory type, with CIs. SMA slope ≈ encoder error
(near zero); the others significantly steeper. Annotate slope ratios.

## Panel C — contradiction rate
Secondary: rate of self-contradiction over the run; SMA's content-addressed,
verifier-gated store should hold near zero while generative memories accumulate
contradictions.

## Inset — the mechanism (TikZ)
Two tiny loops: (left) context-only/RAG — "output → input" autoregressive loop
with an ε error edge that compounds; (right) SMA — "environment → deterministic
encoder → store" loop where the LLM output never re-enters memory. The structural
reason for the flat curve, in one glyph.

**Data sources:** a T5 run via `sma/eval/drift_env.py` → `reports/confirmatory/
t5_*.csv` (drift protocol; Phase 4 deliverable — figure is specced now, data
comes with the drift phase).
**Caption point:** "SMA re-derives working state from the environment each step,
so state-fidelity decays at the encoder's error floor rather than compounding
through generation (A,B), with near-zero self-contradiction (C)."
