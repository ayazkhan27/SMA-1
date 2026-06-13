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
- [x] Phase 2  Scoring resolution: surprisal-SES scorer gauntlet (vs SES, MDL,
      RRF combo) on family strata + EOF case + transfer; ses_n bias study;
      BugsInPy T3 fix-retrieval.
- [x] Phase 3  Freeze train: SSB de-circularization, calibration grid (24
      configs), score-v2 + prereg FROZEN at tag prereg-v1, AND the single-shot
      confirmatory battery (T1-T4 + SSB, seeds 201-205/41,43, paired bootstrap
      + Holm + Cliff's delta). Numbers are now claims (STATUS 2026-06-12).
- [ ] Phase 4  Differentiator (the breakthrough content):
      4a DRIFT (design FROZEN: docs/superpowers/specs/2026-06-12-phase4a-drift-design.md):
         REAL data (no synthetic) on LongMemEval (+ LoCoMo) agent-memory
         benchmark; 4 variants context-only/RAG-notes/Zep-Graphiti(SOTA, run on
         DeepSeek)/SMA; extraction LLM-based & held-constant (isolates the
         memory mechanism); metrics = LongMemEval per-category accuracy
         (knowledge-updates=concept-drift, temporal) + update-recovery/staleness
         + SAGE expectation-violation as a standard drift detector. Supersedes
         blueprint §8.3 synthetic T5.
      4b CROSS-DOMAIN BREADTH + DYNAMIC ADAPTERS (one experiment): expose the
         generic adapters to a real NON-telemetry domain (finance fraud /
         healthcare claims, public + license-clean); MEASURE higher-order-
         relation density and SMA's edge vs structural depth; then run the
         draft-adapter loop (LLM proposes residual HO-relation rules when
         coverage is low) and re-measure whether SMA's structural advantage
         recovers. Validates generality + the dynamic adapter + systematicity-
         as-active-ingredient in one arc. Governance of drafting: see ADR-007.
- [ ] Phase 5  Scale & latency + hardening: real HNSW, tiered WL/SME retrieval,
      parallel FAC, Rust hot loops; gate p95 < 300 ms @ 100k cases. Implement
      ADR-007 adapter RBAC / quarantine / audit / kill-switch.
- [ ] Phase 6  Publish & release: write the manuscript (NeurIPS 2026 primary,
      Nature MI alternative); arXiv preprint; Apache-2.0 (LICENSE present);
      build the clean public tree (scripts/build_release.py -> dist/sma-public),
      GitHub (khanayaz2727@gmail.com), Hugging Face Space (release/hf_space) +
      optional SSB dataset card, Zenodo DOI, Docker artifact. Plan in
      release/RELEASE_PLAN.md.

## Phase 4c — PAPER SPINE: golden-ontology SMA > RAG/KG (2026-06-13)
- [x] Universal OWL/OBO loader + registry + DomainRouter (sma/ontology); loaders
      for OBO, OWL (stdlib + rdflib), multi-file OWL dir, STIX (ATT&CK), CPC,
      MITRE CAPEC/CWE. 11 ontologies / 6 domains loaded for routing (~600k terms):
      medicine HPO+MONDO+GO+Uberon, cyber ATT&CK+CAPEC+CWE, discovery ChEBI+GO,
      legal CPC(254k), finance FIBO. configs/ontologies.json + seed script.
- [x] Pre-registered gigatest (configs/preregistration_ontology.md). De-risk:
      SMA ties the ontology oracle (Phenomizer) but beats real RAG/KG.
- [x] Agentic suite (sma/eval/agentic): memory-swap harness, 6 retrievers incl.
      enterprise RAG (BGE dense / Hybrid-RRF / Hybrid+bge-reranker) + HippoRAG;
      metrics tail top-k + cite-or-abstain AURC + novelty F1. MEDICINE ARM:
      SMA beats best enterprise RAG +33pp tail top-5 (p=0.0002), AURC 0.017 vs
      0.317, novelty F1 0.182 vs 0.000. docs/superpowers/{specs,plans}.
- [x] Agentic arms — 4/4 finished WIN vs best enterprise RAG (tail top-5, Holm-sig):
      medicine +0.333 (p=2e-4), genomics/GO +0.156 (p=2e-4), finance/US-GAAP
      +0.167 (p=2e-4, ~2x best RAG), cyber/ATT&CK +0.073 (p=0.035). SMA best AURC
      + only-nonzero novelty F1 (~0.18) in every arm; pure RAG novelty = 0.
- [x] Legal arm (patent->CPC, USPTO full run): SMA 0.941 vs best RAG 0.870 all-slice,
      +0.064 p=0.002 WIN. 5/5 DOMAINS WIN. Honest: legal rare slice empty (CPC's
      near-uniform IC -> no rare split), reported all-slice + flagged in Fig 2/Table 2.
- [x] Real gold sourced for the 2 hard domains: finance = SEC filings -> US-GAAP
      concepts (FIBO is a schema w/ no instance corpus; honest note); legal =
      USPTO patent -> CPC codes. load_usgaap + arms/{legal,finance}.py.
- [x] Figures: Claude Design conceptual prompts (paper/figures/prompts/: fig_a/b/c
      mechanism+adapter+domains, fig3 capabilities, fig4 boundary; each briefed w/
      science context + Nature MI standard). matplotlib+SciencePlots data figures
      (paper/figures/svg/: figure2_results, figure4_boundary_data, ed1_ssb,
      ed3_ablation) via scripts/figures_paper.py.
- [x] Paper scope agreed (docs/PAPER_SCOPE.md): IN (adapter+agentic 5-domain+
      capabilities+de-risk+flat-tabular boundary+mechanism); SUPPLEMENTARY (SSB +
      transfer -> Extended Data); OUT (general-retrieval battery, exploratory
      rare-disease test, invalid 4a drift, pre-pivot figures). Total figure plan:
      4 main (+1 after LLM-QA) + 4 Extended Data.
- [x] FROZE **adapter-v1** (git tag; ADR-008): universal adapter + agentic harness
      API + pinned ontology versions; 111 tests green, 5/5 validated. figure2/Table2
      refreshed to 5 domains; manuscript recompiled (6pp).
- [x] REGISTERED **prereg-v2** (configs/preregistration_v2_llmqa.md): the LLM-QA
      trustworthy-specialist phase, before any run.

## Phase 5 — LLM-QA "TRUSTWORTHY SPECIALIST" (the breakthrough; prereg-v2)
- [x] Built the memory-swap AGENT (sma/eval/agentic_qa): fixed LLM (DeepSeek
      temp 0) + prompt, swap none/dense-RAG/SMA; one-shot retrieve→{answer,cite,
      abstain}+novelty on HPO diagnosis QA. Calibrated structural cite-or-abstain
      gate (raw grounding score vs a per-memory Youden's-J threshold fit on a
      disjoint split; abstain+flag-novel below it, skipping the LLM call). 48
      tests green; frozen adapter-v1 untouched (commits 99b75bc, c3feed9).
- [x] 3 pools (answerable / held-out=out-of-knowledge≡novel); metrics = accuracy
      + ALCE citation-faithfulness + abstention (risk-coverage + selective acc) +
      threshold-free grounding-AUROC + novelty recall/precision/F1; paired
      bootstrap (10k, seed 12345) + Holm (scripts/agentic_qa_stats.py).
- [x] Pilot → full run DONE (real DeepSeek, n=120+120). RESULT: SMA-grounded
      agent is a VERIFIABLE SPECIALIST — accuracy 0.342 vs dense 0.100 vs
      closed-book 0.017 (Δ+0.242), grounding-AUROC 0.793 vs dense's near-chance
      0.547 (Δ+0.246), novelty-F1 0.789 vs 0.553 (Δ+0.308), selective-acc 0.625
      vs 0.500 (Δ+0.125) — 4/5 axes Holm-significant; abstain-recall a TIE (both
      gates catch ~90% of unknowns, but dense pays 79% false-abstain vs SMA 45%).
      Figure 5 + manuscript (6pp) updated. Nulls reported (anti-cherry-pick).
- [ ] (optional) extend LLM-QA to cyber/legal/finance; interactive AgentClinic
      flagship (single domain).

## Phase 6 — FINALIZE & RELEASE
- [ ] Extended Data: full per-domain metrics table; paired cross-system transfer
      redo (ED1 transfer panel); risk-coverage curves. references.bib polish.
- [ ] (optional) Structural-fraud arm (Elliptic/AMLSim + typology taxonomy) — turn
      the flat-tabular credit-card NULL into a structural WIN (closes the boundary).
- [ ] Manuscript polish -> arXiv preprint -> submission (Nature MI). Apache-2.0
      public tree (scripts/build_release.py -> dist/), GitHub (khanayaz2727@gmail.com),
      Hugging Face Space, Zenodo DOI, Docker. Plan: release/RELEASE_PLAN.md.
