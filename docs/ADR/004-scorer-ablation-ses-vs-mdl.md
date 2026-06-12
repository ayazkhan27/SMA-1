# ADR-004: SES vs MDL scorer ablation (Regime A vs Regime B)

Date: 2026-06-11 · Status: accepted · Relates to blueprint §2.6, §8.7

## Context

A live UI test exposed a retrieval failure: for a query about an EOFException
write-pipeline failure, SMA retrieved anomalous sessions but missed the
EOF *family* entirely, while BM25/dense keyed on the rare exception tokens.
Mechanism: the SES trickle-down scorer weights all matched relations roughly
equally, so abundant common matches (Receiving-block templates, before-chains)
swamp the one rare template that identifies the failure family. Lexical methods
do rare-term weighting implicitly; SES does not. The blueprint anticipates the
fix as Regime B (MDL): rare shared structure compresses more, hence counts more
("surprisal-weighted systematicity"). Ad-hoc IDF-style weights on SES are
forbidden by the no-heuristic-weights mandate.

## Measurements (2026-06-11, LogHub MVP diagnostic, seed 42; EOF test on the
5,000-session HDFS UI corpus, k=5)

| Metric | SES | MDL |
|---|---|---|
| HDFS macro-F1 | **0.9549** | 0.8933 |
| HDFS hit@1 | **0.9550** | 0.8500 |
| BGL macro-F1 | 0.8687 | 0.8687 |
| EOF-family in top-5 (prose query) | 0/5 | **3/5** |
| EOF-family in top-5 (hybrid query) | 0/5 | **2/5** |
| HDFS p50 latency | **354 ms** | 689 ms |

Caveats: the current `mdl_gain` is an MVP (within-target unigram costs over
matched functor *types*, not the blueprint's corpus-level code with pointer and
substitution-table terms). Its rare-family win comes from type-level set
semantics + within-query surprisal; a corpus-cost implementation may close the
aggregate-triage gap and is the natural next iteration.

## Decision

1. SES remains the default scorer (`score-v1` candidate): it wins aggregate
   triage, the headline within-system claim.
2. MDL is kept as a first-class, user-visible alternative: exposed as a scorer
   toggle in the comparison UI and reported alongside SES in evals
   (reports/triage_metrics_mdl.csv). The rare-family result is a finding, not
   an implementation detail.
3. No IDF-style patching of SES — any rarity weighting must come from the MDL
   code-length formulation (upgrade path: corpus-level functor costs supplied
   by the index to the scorer).
4. Cross-system transfer (T2-b) runs under BOTH scorers; if MDL's advantage is
   real it should be largest there.

## Consequences

- The scorer choice stays open until the §8.6 calibration freeze; this ADR
  records the evidence available at decision time.
- Known SES limitation (rare-template blindness) is documented for the paper's
  limitations section regardless of the final default.
