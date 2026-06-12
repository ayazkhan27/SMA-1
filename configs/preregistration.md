# SMA-1 Preregistration

Status: **FROZEN at git tag `prereg-v1`** (2026-06-12). Everything in this
document was fixed before any confirmatory test run listed in section 4 was
executed. Numbers observed during development (docs/STATUS.md ledger) are
exploratory; only post-tag runs are reported as confirmatory results.

## 1. Frozen score function (score-v2-final)

Selected on validation data only — seed 7 HDFS sample (test protocol uses
fresh seeds), SSB validation seeds 29/31 (fixtures use 11/19/23). Full grid:
`reports/calibration_grid.csv` (24 configs, 3 metrics each).

| dial | frozen value | evidence (validation) |
|---|---|---|
| scorer | `surprisal` (score-v2, ADR-005) | family-hit common 0.8491 vs 0.8415 (ses); ties elsewhere at rho=0.95 |
| normalization | `max` | beats `target` on family (0.849 vs 0.800) and LOO-haystack (0.92 vs 0.90) |
| gamma | 0.25 | flat across 0.125/0.25/0.5 under the winning config; blueprint default kept |
| rho | 0.95 | the ONLY validation failures in the grid are rho=0.90 + surprisal/max (SSB r1 0.79); 0.95 is 1.00 everywhere |
| delta | 0 (single-ontology adapters); 2 when an adapter declares a predicate lattice (SSB) | de-circularized SSB requires lattice ascension by construction (ADR-006) |

These equal the `MatchConfig` defaults in `sma/match/types.py` at the tag.
After the tag these dials do not move; any change requires a new tag and a
fresh test battery.

Registered caveat: the haystack validation probe is leave-one-out (needle
queries drawn from the corpus). Out-of-corpus needle queries under solo SMA
with max-normalization underperformed during development; the registered
production posture for haystack workloads is hybrid fused (RRF of
BM25+dense+SMA with SME alignment receipts), reported alongside solo modes.

## 2. Hypotheses

- **H1 (cross-system transfer, primary):** SMA retrieval transfers across
  log systems sharing only the frozen ontology (`ontology-v1`, no per-target
  tuning) with macro-F1 exceeding BM25 and dense baselines.
- **H2 (structure vs surface):** on the de-circularized SSB (zero lexical
  overlap, lattice-only bridging), SMA ranks the structural analog first
  while lexical baselines fail (r1 at chance or below).
- **H3 (grounded generation):** answers produced from SMA evidence with the
  cite-or-abstain policy contain fewer invented entities than context-only
  prompting (judged pass; protocol in scripts/h3_mini_study.py).
- **H4 (rare-event leverage):** surprisal weighting improves family-hit on
  rare failure families without degrading common ones vs unweighted SES.
- **H5 (cross-domain reach):** the same frozen matcher beats lexical/dense
  baselines on BugsInPy LOPO category@1 (code domain, no log vocabulary).
- **H6 (drift):** under concept drift (T5 protocol), SAGE
  expectation-violation flags drifted behaviour earlier than frequency
  baselines (wrong-action rate).

## 3. Datasets and leakage discipline

HDFS_v1, BGL, Spirit, Thunderbird, OpenStack, Liberty (LogHub, md5-verified
manifests in data/manifests/), BugsInPy. Label columns are stripped before
encoding (BGL/Thunderbird/Spirit alert column; OpenStack source filename);
family labels derive from session text via `sma/eval/family_labels.py` only.
Ontology frozen at tag `ontology-v1` (hash fd345c5) before Spirit was first
read.

## 4. Confirmatory test protocol (single shot, post-tag)

Test seeds: **{201, 202, 203, 204, 205}** — never used during development or
calibration. SSB test seeds: **{41, 43}**, n=100 library. One execution per
cell; no reruns except for crashes (crash reruns logged in STATUS.md).

- **T1 transfer:** BGL→Spirit, BGL→Thunderbird, HDFS→OpenStack; index 800 /
  query 200 per seed; macro-F1, label-hit@1.
- **T2 within-system:** HDFS family-hit@5 (common/rare strata), BGL triage;
  1000-session stratified samples.
- **T3 code:** BugsInPy LOPO category@1.
- **T4 haystack:** Liberty 5000 @ 5% needles, out-of-corpus needle probes;
  hybrid fused primary, solo modes reported.
- **SSB:** forced-choice and library r1/MRR vs BM25/dense.
- **Baselines:** BM25, TF-IDF dense, real-embedding dense, hybrid RRF
  (+rerank), HippoRAG-2-style KG comparator. Baseline implementation details
  are independent of the frozen SMA dials and may be completed post-tag;
  baseline hyperparameters get the same validation-only discipline.

## 5. Statistics

Per-query paired bootstrap (10,000 resamples) for SMA-vs-baseline deltas
with 95% CIs; Holm-Bonferroni correction within each dataset's family of
baseline comparisons; Cliff's delta reported as effect size; multi-seed
means ± s.d. across the five test seeds. A hypothesis is supported only if
the corrected CI excludes zero in the registered direction.
