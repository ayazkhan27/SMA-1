# Figure 3 — Certified MAC/FAC retrieval (the algorithmic contribution)

**Role:** prove Claim C1 — that the content-vector bound is *admissible* and buys
exact early stopping. This is the figure that makes the retrieval a contribution,
not just plumbing. No MAMMAL analog (our novelty); model the rigor on a good
algorithms-paper panel.

**Toolchain:** Matplotlib (SciencePlots), CMasher for the density coloring,
**brokenaxes** for the score-scale gap, adjustText for the annotated points.

## Panel A — the bound is admissible (scatter)
x = true SESₙ (full SME), y = U-bound (histogram intersection × constant), one
point per (query, candidate) pair from a 1k sample. The diagonal y=x drawn; the
admissibility claim is **every point on or above the diagonal** (bound never
underestimates). Points colored by CMasher density. A shaded "violation zone"
below the diagonal stays provably empty — that emptiness IS the lemma, shown.
Inset: histogram of the slack (U − SES), always ≥ 0.

## Panel B — best-first early stop (the mechanism)
For one representative query: candidates sorted by descending bound on the x
axis; two step curves — the running k-th-best SESₙ (solid teal) and the bound of
the next unexamined candidate (dotted). The **crossing point** where
`bound(next) < kth-best` is starred and labeled "certified stop — everything to
the right provably cannot enter top-k." Shade the skipped region (compute saved).
A side number: "examined 31 of 1600 candidates; top-k provably exact."

## Panel C — shortlist recall + latency (the gate)
Small twin bars / line: recall@shortlist vs brute force (target ≥ 0.98, gate G4)
and query p95 latency vs corpus size N (log x), with the 20k ANN_THRESHOLD
crossover marked where bound-ordering hands off to ANN pre-screen. Honest note if
any cell is below target.

**Data sources:** a dedicated `reports/macfac_certification.csv` (to be produced
by a diagnostic run over a 1k sample — `MacFacIndex.retrieve` vs `brute_force`).
**Caption point:** "The MAC content-vector bound is admissible (A), so best-first
FAC returns a provably exact top-k after examining a small fraction of the
library (B), at sub-second p95 to 10⁴ cases (C)."
