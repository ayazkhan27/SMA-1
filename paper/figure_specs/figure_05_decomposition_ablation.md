# Figure 5 — Decomposition & ablation (what is the active ingredient?)

**Role:** internal-validity figure — isolate *why* SMA works, pre-empting the
reviewer's "is it just the representation?" Small-multiples grid. Analogous to
MAMMAL's per-benchmark ablation reporting, but visual.

**Toolchain:** Matplotlib small-multiples (SciencePlots `science` style),
shared y-axis, CMasher for the gradient sweeps.

## Panel A — representation vs alignment (the active-ingredient ladder)
Horizontal ladder of macro-F1 (one transfer leg): raw text BM25 → WL-1 content
vectors only (representation, no alignment) → MAC bound only → full SMA (MAC+SME
alignment). The jump from "WL-1 only" to "full SMA" is the **SME alignment
contribution** — annotate that delta explicitly. Compare against Hybrid+Rerank
on the same axis to show alignment beats a strong learned reranker.

## Panel B — γ sweep (systematicity dial)
Line: macro-F1 / SSB-r1 vs γ ∈ {0, 0.125, 0.25, 0.5}. γ=0 kills trickle-down →
expect far-analogy collapse (the internal control). The frozen γ=0.25 marked
with a vertical guide. Shows the result is not knife-edge sensitive.

## Panel C — scorer & normalization
Grouped bars: SES vs surprisal vs MDL across {family-common, family-rare, EOF,
transfer} — the matrix that motivated freezing surprisal/max. From the
calibration grid. Show the rare-family lift of surprisal (the H4 mechanism).

## Panel D — encoder-noise robustness
Degradation curve: perturb 10–40% of statements, plot SMA SESₙ retention vs
baselines' score retention. Structure degrades gracefully; this is risk-register
R2 answered with data.

**Data sources:** `reports/calibration_grid.csv` (B, C), a dedicated
decomposition CSV (A) from `transfer_eval` with method ablations, a perturbation
diagnostic (D). 
**Caption point:** "SME alignment — not representation alone — is the active
ingredient (A); the systematicity dial behaves as theory predicts (B); surprisal
weighting earns its keep on rare failure families (C); and structural scores
degrade gracefully under encoder noise (D)."
