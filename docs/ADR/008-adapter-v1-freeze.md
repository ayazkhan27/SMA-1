# ADR-008 — Freeze `adapter-v1` (universal ontology adapter + agentic harness)

Status: **ACCEPTED / FROZEN** (2026-06-13, git tag `adapter-v1`).
Supersedes nothing; complements `prereg-v1` (the matcher dials, ADR-005/006).

## Context

The golden-ontology paper spine (`docs/PAPER_SPINE.md`) is validated: SMA beats
the enterprise RAG/KG gauntlet on the tail across **five unrelated domains**
(medicine, finance, genomics, cyber, legal — all Holm-significant; `docs/STATUS.md`).
The code that produced this — the universal ontology adapter and the memory-swap
agentic harness — is stable and must be pinned so every reported number is
reproducible and the next phase (LLM-QA, `prereg-v2`) builds on a fixed substrate.

## Decision — what is frozen

**Frozen public API (no signature/semantics change without a new tag):**

- `sma/ontology/`:
  - `OntologyGraph`, `Term` (the normalized representation).
  - Loaders: `load_obo`, `load_owl`, `load_owl_dir`, `load_rdflib`,
    `load_attack_stix`, `load_cpc`, `load_mitre_xml`, `load_usgaap`,
    `load_ontology`, `fid`.
  - `mount() -> MountedOntology` (`.canon`, `.config`, `.build_case`, `.build_index`).
  - `OntologyRegistry`, `OntologyEntry`, `DomainRouter`.
- `sma/eval/agentic/`:
  - `Memory` protocol + `IndexItem`, `Query`, `Retrieved`.
  - The six memories: `SmaMemory`, `BM25Memory`, `DenseMemory`,
    `HybridRRFMemory`, `HybridRerankMemory`, `HippoMemory`.
  - `run_oneshot(...)`; metrics `tail_topk`, `risk_coverage_aurc`, `novelty_f1`.
- Matcher dials remain at `prereg-v1`: `MatchConfig(scorer="surprisal",
  normalization="max", gamma=0.25, rho=0.95, delta=2)`.

**Pinned ontology versions** (`configs/ontologies.json`, md5/versions in
`data/manifests/datasets.json`): HPO `hp/releases/2026-06-06`, MONDO `2026-06-02`,
GO `2026-05-19`, Uberon `2026-04-01`, ChEBI `chebi/252`, ATT&CK `2.0`,
CAPEC `3.9`, CWE `4.20`, CPC `2026-05-01`, FIBO (rdflib subset), US-GAAP `2024`.

**Frozen evaluation protocol:** memory-swap (LLM/prompt fixed, retriever varies);
hard partial/imprecise/noisy queries; rare slice = entity's rarest-term
IC > corpus median (with the documented degeneracy on near-uniform-IC ontologies
like CPC → all-query slice, ADR note); paired bootstrap (10k, seed 12345) + Holm;
`PYTHONHASHSEED=0` + sorted set→list.

## Validation at freeze

- `tests/` (ontology + agentic): 111 passed, 1 slow skipped.
- 5/5 agentic arms WIN vs best enterprise RAG on tail top-5 (Holm-sig):
  medicine +0.333, finance +0.167, genomics +0.156, cyber +0.073, legal +0.064.
- SMA best risk-coverage AURC + only non-zero novelty F1 in every arm.

## Policy

- After `adapter-v1`, these interfaces and the ontology versions **do not move**.
  Bug fixes that do not change API or numeric semantics are allowed on the same
  tag; any API/semantics/ontology-version change requires a new tag + a fresh
  validation run.
- New ontologies/domains may be **added** (they don't alter frozen behaviour).
- The LLM-QA phase (`prereg-v2`) consumes this frozen adapter unchanged.

## Known limitations recorded at freeze

- Rare-slice definition degenerates on near-uniform-IC ontologies (CPC/legal) →
  reported on the all-query slice; a percentile/frequency-based rare slice is a
  candidate revision (would need a cross-domain re-run for consistency).
- HippoRAG baseline is disadvantaged by the bag-of-term-names document
  representation; novelty-F1 threshold is fixed at 0.5 (untuned).
- Cross-system transfer figure (ED) needs a paired per-leg redo (not leg-averaged).
