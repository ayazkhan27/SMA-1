# ADR-006: Matcher semantics v4 (the Liberty forensic arc)

Date: 2026-06-13 · Status: accepted · Trigger: a user-commissioned
needle-in-haystack test (Liberty, 5,000 sessions @ 5% admin-alert needles)
on which solo-SMA scored 0/5 while BM25/dense scored 5/5.

## Defects found and fixed (each now a regression test)

1. Vacuous constant pairing: event_type/integer entities paired as variables,
   letting count(A,3) match count(B,2). event_type entities are now constants
   (match identically); integers remain variables (same-template different-
   count IS analogous; releasing them changed nothing empirically).
2. Parallel-connectivity violation: support closure induced argument
   statement pairs without functor-compatibility checks, letting `before`
   matches manufacture cross-template correspondences that surprisal weights
   amplified (a 6-statement match outscoring a 14-statement self-map;
   ses_n > 1 under min-norm). Argument correspondences now require canonical
   identity or lattice ascension within delta at rho^dist penalty; illegal
   pairs invalidate the kernel (SME's actual rule).
3. MAC mis-gating: the ANN cosine pre-screen (kept 1/250 needles in top-200)
   ran BEFORE the admissible bound ordering (which had 21/30 right). For
   corpora <= 20k cases the weighted Lemma-2 bound now orders all candidates
   directly; ANN is a >20k scale optimization only.
4. Normalization is not a constant: measured on the two adversarial probes,
   max-norm wins transfer (0.899) and fails haystacks (0/5); min/target-norm
   the reverse (5/5 / 0.344). sqrt eliminated. FINDING: no single scale-free
   form serves both precision-like (class discrimination) and recall-like
   (size-asymmetric needle) demands. Normalization joins gamma/rho/delta/
   theta as a calibrated, task-selected parameter. Default: "max".

## Citable numbers under v4 (seed-42 protocol, reports/*_v4final.csv)

| Metric | Value |
|---|---|
| BGL->Spirit transfer (surprisal) | macro-F1 0.8994, hit@1 0.9050 (dense 0.31, BM25 0.40) |
| HDFS family-hit@5 common / rare (surprisal) | 0.728 / 0.850 |
| BGL family-hit@5 common (surprisal) | 0.8705 |
| EOF micro-case | ses 3/5 (scorer default re-decision deferred to calibration) |
| Size-bias indicator | eliminated (18.2 vs 18.2) |
| Liberty haystack, solo SMA / hybrid | 0/5 (max-norm, documented) / 5/5 via hybrid receipts |

## Consequences

- HDFS common-family deflation (0.89 -> 0.73) is honest: the delta was
  vacuous-match score and shortlist luck. All paper assets regenerate from
  v4final CSVs; pre-v4 figures were hatched and are now replaced.
- The scorer question (ses vs surprisal: EOF flip vs rare-strata win) and the
  normalization selection are calibration-phase decisions, pre-registered as
  such.
