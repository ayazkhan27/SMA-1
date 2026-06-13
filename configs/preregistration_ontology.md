# Preregistration — Multi-Domain Ontology Benchmark Suite ("the gigatest")

Status: **REGISTERED 2026-06-13, before any confirmatory arm below is run.**
Companion to `configs/preregistration.md` (the core battery) and the thesis in
`docs/PAPER_SPINE.md`. This document fixes the domains, gold tasks, baselines,
metrics, and per-arm falsifiers **in advance**, so the suite is a test of the
golden-ontology thesis and not a post-hoc trophy case.

## 0. Anti-cherry-pick rule (the whole point of registering)

1. Every arm listed in §3 is reported in the paper **whether it wins, ties, or
   loses** — exactly as the 4b flat-tabular null is reported.
2. Any domain added *after* this registration is marked **exploratory** and may
   not be reported as confirmatory without a new dated registration entry.
3. No arm's metric, baseline, or held-out protocol may change after its first
   confirmatory run. A change requires a new registration + a fresh run.
4. The arm list is frozen at the count below; we do **not** keep adding domains
   until enough win. Power comes from breadth of the *recipe*, not selection.

## 1. Claims under test (from PAPER_SPINE §6)

- **C1 (rarity):** on a golden-ontology domain, SMA beats the vector/IC baseline
  on top-k for k≥5 at rare/long-tail queries.
- **C3 (across fields):** the **same** mounting recipe (`load_obo → mount →
  build_index → retrieve`, no per-domain code) wins in **≥2 unrelated**
  golden-ontology domains.

The suite confirms C3 iff ≥2 arms confirm C1 with no code specialized per domain.

## 2. Shared protocol (identical across all arms)

- **Pipeline:** `sma.ontology.load_obo(...) → mount(graph) → build_index(records)
  → retrieve(query, k=10, shortlist=80, fac_budget=40)`. The mount config is the
  frozen `MatchConfig(delta=2, rho=0.95, scorer="surprisal", normalization="max")`
  — identical dials to `prereg-v1`. **No per-domain encoder.**
- **Records:** each domain entity (disease, protein, …) is one case: its
  annotation term-set, each term a FUNCTOR over a constant subject, the
  ontology's is-a tree as the ascension lattice, typed relations lifted to
  higher-order statements (where the ontology has them).
- **Query (held-out, hard):** partial + imprecise observation — sample
  `min(5, |terms|)` of an entity's terms, climb 0–2 is-a levels (imprecision),
  add 3 noise terms; retrieve and check the true entity's rank.
- **Baselines:** (a) **Phenomizer/Resnik IC best-match semantic similarity** —
  the ontology-aware SOTA-equivalent; (b) **Jaccard** term-overlap — a lexical
  floor. Same query set scored by all three.
- **Reproducibility:** ontologies version-pinned by md5 in
  `data/manifests/datasets.json` (`obo_foundry`, `hpo`); runs set
  `PYTHONHASHSEED=0` and sort every set→list conversion (fixes the run-to-run
  variance seen in the exploratory HPO gate). Seeds {7, 17, 23}.
- **Primary metric:** top-5 accuracy. **Secondary:** top-10, top-1, MRR.
- **Statistics:** per arm, paired bootstrap (10k, seed 12345) on per-query
  correctness for Δ(SMA − best baseline) on the primary metric →
  {delta, 95% CI, p}; **Holm–Bonferroni across the family of arms**; Cliff's δ
  effect size. Reuses `sma/eval/stats.py`.

## 3. The registered arms

| # | Domain | Ontology (pinned) | Entities / gold | Baseline | Status |
|---|---|---|---|---|---|
| **A1** | Rare-disease diagnosis | HPO `hp/releases/2026-06-06` | diseases ← `phenotype.hpoa` (aspect P, 7–30 phenotypes) | Phenomizer IC | **confirmatory pending** (exploratory: SMA wins top-5/10 across seeds) |
| **A2** | Gene function | GO `releases/2026-05-19` (go-basic) | proteins ← `goa_human.gaf` (aspect P/BP, 7–30 GO terms) | Resnik IC best-match | **confirmatory pending** |
| **A3** | Disease cross-vocabulary | MONDO `releases/2026-06-02` | MONDO classes ← xref/synonym sets | IC similarity | registered, **task spec pending** |
| **A4** | Cyber TTP retrieval | MITRE ATT&CK (STIX) | incidents ← technique sets | KG/embedding retrieval | registered, **data drop pending** |

A1 and A2 are runnable now (data on disk). A3/A4 are registered with their
ontology + intended gold; their confirmatory run is gated on the task corpus and
will follow the identical protocol. The C3 "across fields" claim is earned by
A1 (medicine) + A2 (biology) being unrelated golden ontologies under one recipe.

## 4. Per-arm hypotheses & falsifiers

- **H-A1 / H-A2 (primary):** SMA top-5 > best baseline top-5.
  *Falsifier:* paired-bootstrap 95% CI for Δtop-5 includes 0 after Holm → the arm
  is reported as a **null/parity** (not a win), like 4b.
- **H-suite (C3):** ≥2 arms confirm H primary.
  *Falsifier:* fewer than 2 arms survive Holm → the "across fields" claim is
  **dropped** and the paper narrows to the single confirmed domain.

## 5. Registered caveats

- A2's IC baseline uses GO term frequencies from the same GAF (closure-propagated),
  matching A1's Phenomizer construction; no protein sequence / BLAST baseline is
  claimed (out of scope — this is an *ontology-structure* benchmark, not sequence
  homology).
- Term-set rarity differs by domain; we report the per-arm rare-slice (entities
  whose rarest term has IC above the corpus median) separately, as C1 is about
  the long tail specifically.
- These are **retrieval** arms (rank the true entity). Downstream LLM generation
  (cite-or-abstain) is the core battery's H3, not re-tested here.
