# /goal SMA-MVP-1

MISSION: Build and evaluate Structure-Mapping Agentic Memory exactly per blueprint sections 2-8.

DEFINITION OF DONE:

- D1. `pytest -m "gate_G0 or gate_G1 or gate_G2 or gate_G3 or gate_G4 or gate_G5 or gate_G6"` passes.
- D2. Canonical battery includes water-flow/heat-flow and solar-system/atom mappings, plus SME v4 oracle fixtures.
- D3. Certified retrieval verified: best-first FAC top-k equals brute-force top-k on a fixed sample.
- D4. Preregistration exists and is frozen before test-set runs.
- D5. Full evaluation emits `reports/report.html` from `make report`.
- D6. Every agent-surfaced claim in the demo carries provenance or is explicitly unsupported.
- D7. H1-H6 are reported whether positive or negative.

STOP CONDITIONS: D1-D7 satisfied, or a blueprint kill criterion fires and the prescribed write-up is delivered.


## Phase ledger (post-MVP execution plan, agreed 2026-06-12)

- [x] Phase 0  Protect & freeze: git history, CI, ontology-v1 tag, positioning
- [x] Phase 1  Credibility batch: held-out Spirit transfer (3 seeds, frozen
      ontology, SMA 0.938 vs dense 0.356), production-RAG ladder (+B6, WL
      control; ColBERT skipped as stretch), family-hit@k metric, H3 LLM-judge.
- [ ] Phase 2  Scoring resolution: surprisal-SES scorer gauntlet (vs SES, MDL,
      RRF combo) on family strata + EOF case + transfer; ses_n bias study;
      BugsInPy T3 fix-retrieval.
- [ ] Phase 3  Freeze train: SSB de-circularization; calibration on validation;
      freeze score-v2; prereg git tag; single-shot test runs with paired
      bootstrap CIs (numbers become claims).
- [ ] Phase 4  Differentiator: drift protocol T5 (wrong-action rate under
      concept drift); OpenStack missing-event spike (SAGE expectation
      violation).
- [ ] Phase 5  Scale & latency: real HNSW, tiered WL/SME retrieval, parallel
      FAC, Rust hot loops; gate p95 < 300 ms @ 100k cases.
- [ ] Phase 6  Publish & pilot: paper, arXiv preprint, Apache-2.0 release,
      Docker artifact, stakeholder demo pack.
