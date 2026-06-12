# Figure 6 — SSB: structure beats surface under perfect ground truth (H2)

**Role:** the decisive controlled instrument. The Synthetic Structural Benchmark
orthogonalizes vocabulary from structure, so the win is unconfounded. This figure
should make the orthogonalization *visible*.

**Toolchain:** TikZ mini-DAG insets (the triple geometry) + Matplotlib bars
(the outcome). CMasher for the 2×2 heat tint.

## Panel A — the triple, drawn (TikZ insets)
Three small DAGs side by side for one SSB item:
- **Query** Q — a relational schema, vocabulary 1.
- **Analog** A(Q) — same skeleton, **vocabulary 2 (zero overlap)**, bridged by
  the generated lattice (dotted violet, as in Fig 2). Tag "same structure,
  different words."
- **Distractor** D(Q) — **vocabulary 1 (identical words)** but **star-rewired**
  so the justification DAG is broken. Tag "same words, broken structure."
The hub node of the star distractor circled to show non-isomorphism to the chain.

## Panel B — the 2×2 orthogonalization grid
A 2×2: rows = {near vocabulary, far vocabulary}, cols = {intact structure,
scrambled structure}. Each cell a small bar of "fraction ranking the true analog
first" for SMA vs best lexical baseline. The diagonal tells the story: far-vocab
+ intact-structure is where lexical methods rank the *distractor* first (their
similarity IS the matched histogram) and SMA ranks the *analog* first. CMasher
tint by SMA advantage.

## Panel C — forced-choice & library r1/MRR
Clean bars: SMA r1=1.0 (both seeds) vs BM25/dense ≈ 0.0 on forced-choice and the
24-case library, with the Holm-corrected p. The "answer-key-free" caption note:
the de-circularized generator (disjoint lexicons, lattice-only bridge, star
distractors) means this is honest.

**Data sources:** `reports/confirmatory/ssb_summary.csv`, `ssb_stats.csv`,
`ssb_rows.csv`; the triple geometry from `ssb_generator.generate_triples`.
**Caption point:** "When vocabulary and structure are orthogonalized under
ground truth (A,B), lexical similarity tracks the surface distractor while
structure mapping tracks the true analog (C) — the conjecture isolated."
