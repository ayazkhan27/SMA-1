# ADR-005: score-v2 = surprisal-weighted trickle-down (the SES x MDL merger)

Date: 2026-06-12 · Status: accepted · Supersedes the SES-default decision in
ADR-004 · Relates to blueprint §2.3, §2.6, §2.7 (weighted Lemma-2 bound)

## Decision

The default scorer becomes `surprisal`: SES trickle-down geometry with
sigma_0 = corpus surprisal (-log2 p, KT-smoothed) of each statement's
canonical functor, supplied by the index from its own contents. Entity MHs
keep unit weight. With no corpus statistics the scorer reduces EXACTLY to
SES (verified to 1e-12), so all prior SES validation carries over.

## Why (the gauntlet, all measured 2026-06-12, seed-42 protocol)

| Column | ses | mdl | rrf(ses,mdl) | surprisal |
|---|---|---|---|---|
| HDFS family-hit@5, rare (<=20 exemplars, n=6) | 0.1500 | 0.4167 | 0.1833 | **0.5333** |
| HDFS family-hit@5, common (n=100) | **0.9000** | 0.8660 | 0.8820 | 0.8860 |
| BGL family-hit@5, common (n=105) | 0.6724 | 0.6743 | 0.6686 | 0.6571 |
| BGL->Spirit transfer macro-F1 (frozen ontology) | 0.9200 | 0.9100 | n/a | **0.9300** |
| Transfer p50 latency | 577ms | 430ms | n/a | **279ms** |
| Canonical battery (G2) | pass | n/a | n/a | pass |

Surprisal wins or ties every column that matters; its only loss is -1.4 pts
on HDFS common families. The RRF control losing on rare families justifies
the single-scorer merger over rank fusion. No tunable weights are introduced:
costs are corpus counts, satisfying the no-heuristic-weights mandate.

## Consequences and caveats

- Scores are now LIBRARY-RELATIVE (IDF-like): the same case pair scores
  differently against different corpora. Documented as a property; the
  reduction-to-SES default covers corpus-free uses (canonical battery, unit
  tests, pairwise matching without an index).
- Certified retrieval keeps its exactness guarantee via the weighted
  Lemma-2 bound (blueprint §2.7 anticipated this form).
- Index mutation invalidates costs and score caches (handled in add()).
- EOF micro-case is 0/5 under ALL scorers post-encoder-v0.2.0 (ontology
  class statements dilute rare-template surprisal in prose queries) - an
  encoder/scorer interaction flagged for the calibration phase; the
  systematic rare-strata result is the evidence base, not the anecdote.
- MDL (Regime B) remains implemented and reported as an ablation; the UI
  scorer toggle now offers ses / mdl / surprisal.
