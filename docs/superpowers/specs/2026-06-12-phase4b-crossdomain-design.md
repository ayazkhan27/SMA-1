# Phase 4b — Cross-domain breadth + dynamic-adapter validation (design)

Date: 2026-06-12 · Status: building the "before" half in parallel with the 4a run · Decision basis: brainstorm (test-as-is, then draft) + dataset pick (Diabetes-130 healthcare first, IEEE-CIS finance second).

## 1. Goal & hypothesis

Test the generality claim AND the dynamic-adapter capability in one honest arc,
on real non-telemetry domains:
- **H7 (generality / systematicity is the active ingredient):** with the GENERIC
  `structured` adapter (flat, HO-relation density ≈ 0), SMA is at PARITY with
  lexical/dense baselines — because flat tabular data is structurally identical
  across rows, so systematicity has nothing to exploit.
- **H8 (the memory writes its own encoder):** when the LLM drafts residual
  higher-order rules for the domain (raising HO-density), SMA's structural
  retrieval IMPROVES and pulls ahead of the baselines — recovering the advantage
  exactly where structure appears.

This isolates systematicity as the active ingredient and validates the
dynamic-adapter loop as a real capability (currently only unit-tested), under
the ADR-007 governance posture (admin-gated drafting).

## 2. Why flat tabular makes the point (confirmed empirically 2026-06-12)

A real Diabetes-130 encounter encoded by the generic `structured` adapter:
88 statements, **HO-density = 0.000** — pure first-order `(attribute row value)`
triples. Because every encounter shares the identical 50-column schema, the
MAC content vector (functor counts) is ~constant across encounters → no
structural signal; only values vary (the baselines' home turf). This is the
honest "before."

## 3. Datasets (real, public, license-clean)

- **Healthcare (primary):** UCI Diabetes 130-US hospitals (CC BY 4.0), 101,766
  encounters, 50 cols, ICD-9 diagnoses (diag_1..3), ~23 medication columns,
  readmission target {NO, >30, <30}. md5 in `data/manifests/datasets.json`.
- **Finance (second, for breadth):** IEEE-CIS Fraud (transactions). Queued.

## 4. Task & metrics

- **Task:** readmission prediction by analogy. Binarize target to `<30` (early
  readmission, the clinically important positive) vs not. For each query
  encounter, retrieve top-k analogous encounters from an index, predict by
  label vote. Train/test split (or LOPO by patient) with fixed seeds.
- **Methods:** SMA (MacFacIndex over encoded cases) vs BM25 vs dense, all over
  the SAME encoded artifacts where applicable.
- **Metrics:** macro-F1 + readmission-hit@k; HO-relation density of the encoding
  (the mechanism variable); paired bootstrap + Holm + Cliff's δ (reuse stats).
- **Outcome reported honestly** (parity expected for "before"; improvement is
  the "after" claim).

## 5. The two-phase arc

1. **BEFORE (deterministic, no LLM — built in parallel with 4a):** sample
   encounters, encode with the generic `structured` adapter, measure HO-density
   (≈0) and SMA-vs-baseline (expect parity). `scripts/crossdomain_before.py` →
   `reports/confirmatory/cd_diabetes_before.csv`.
2. **AFTER (small LLM burst, deferred until 4a frees the DeepSeek budget):** run
   the draft-adapter loop (`sma/agent/adapter_draft.py`) — the LLM proposes
   residual HO-relation rules for the domain (e.g. `treats(med, diag)`,
   `comorbid(diag_i, diag_j)`, `escalates(change, readmission)`); apply via the
   frozen-keyword-dedup `DraftAdapter`; re-encode; re-measure HO-density (now >0)
   and SMA-vs-baseline (expect SMA gains). Drafting is admin-gated per ADR-007.

## 6. Components

- `data/manifests/datasets.json` — diabetes130 entry (done).
- `sma/eval/diabetes.py` — loader/sampler + binarized label + per-encounter
  artifact builder (flat CSV-row for the generic adapter).
- `scripts/crossdomain_before.py` — the deterministic before-measurement.
- (AFTER) reuse `sma/encoders/draft_adapter.py`, `adapter_draft.py`,
  `coverage.py`; `scripts/crossdomain_after.py`.
- Reuse `sma/index/macfac.py`, `sma/eval/baselines/*`, `sma/eval/stats.py`.

## 7. Honest caveats

- The generic adapter is weak on tabular (schema-identical rows) — that is the
  point, not a bug; it motivates the drafted adapter.
- "After" uses an LLM to DRAFT rules (not to encode) — encoding stays
  deterministic; the drafted rules are data, content-addressed, "unreviewed"-
  tainted (ADR-007). The claim is "the LLM extends the encoder," not "the LLM
  encodes."
- SMA may stay at parity even after drafting if the drafted structure is weak —
  reported honestly.

## 8. Out of scope (YAGNI)

- No hand-built domain encoder (would void the dynamic-adapter thesis).
- Finance (IEEE-CIS) is the second domain, after healthcare lands.
