# Paper scope — what's IN the Nature MI paper vs exploratory / earlier-phase

> Status: **proposal for agreement (2026-06-13).** The repo accumulated work
> across a pivot: an earlier "SMA as a general structural retriever" line
> (Phase 3, `prereg-v1`) and the current **golden-ontology + agentic** thesis
> (Phase 4c). This doc draws the line so the paper is focused and every reported
> number is confirmatory, not exploratory. Three buckets: **IN**, **SUPPLEMENTARY**
> (appendix / Extended Data), **OUT** (exploratory, superseded, or a separate paper).

## The paper's single thesis
A structure-mapping memory grounded in a *golden domain ontology*, delivered by a
*universal adapter*, turns a generalist LLM into a **verifiable specialist** that
beats enterprise RAG/KG on the rare/tail slice across fields — with provenance,
calibrated abstention, and novelty detection RAG cannot provide.

## IN — the paper's contributions and confirmatory results

| Item | Where | Role |
|---|---|---|
| **Universal ontology adapter** (loader/registry/router; OBO/OWL/TTL/STIX/CPC/MITRE-XML/US-GAAP) | `sma/ontology/` | Contribution 1 (system). Fig 1b, Table 1 (11 ontologies / 6 domains). |
| **Agentic memory-swap benchmark, 5 domains** (medicine, genomics, finance, cyber, legal) — SMA vs BM25 / BGE-dense / hybrid-RRF / hybrid+rerank / HippoRAG | `sma/eval/agentic/`, `scripts/agentic_suite.py` | Contribution 2 + **headline result**. Fig 1a/2, Table 2. |
| **Capability metrics** — tail top-5, cite-or-abstain risk-coverage AURC, structural-novelty F1 | `sma/eval/agentic/metrics.py` | The "what RAG can't do" axes. Fig 2b/2c. |
| **Pre-registered de-risk retrieval suite** (SMA ties Phenomizer IC oracle; beats Dense/Hybrid/HippoRAG) | `sma/eval/ontology_bench.py`, `configs/preregistration_ontology.md` | Frames the claim precisely (parity vs oracle, win vs RAG/KG). Extended Data. |
| **The honest boundary — flat-tabular null** (readmission, card-fraud → parity; no ontology ⇒ no SMA advantage) + the tail flat-vs-structural definition | Phase 4b results, `docs/STATUS.md` | Delimits the thesis (Methods + Discussion). The credit-card-fraud boundary example. |
| **SMA mechanism** — SME/MAC-FAC, surprisal (IC) weighting, lattice ascension ρ^d, SAGE expectation-violation | `sma/match/`, `sma/sage/` | Methods. Fig 1a. Frozen dials `prereg-v1`. |

## SUPPLEMENTARY — appendix / Extended Data (supports, not headline)

- **Phase 3 mechanism validations** that the matcher works as claimed: the
  de-circularized **SSB** (zero lexical overlap, lattice-only bridging) and
  **cross-system transfer** (log systems sharing only the ontology). `prereg-v1`,
  `reports/confirmatory/{ssb_*,t1_*}.csv`. → Extended Data: "the structure-mapping
  engine, validated in isolation." (Not the golden-ontology headline, but it shows
  the engine is sound.)
- **MAC/FAC admissible bound (Lemma 2)** + complexity — Methods/appendix.
- Per-domain ontology details + routing demo — Supplementary.

## OUT — exploratory, superseded, or a separate paper (NOT reported as results)

- **Phase 3 application battery framed as a *general* retrieval paper** — rare-family
  retrieval, **BugsInPy** code, triage/haystack, OpenStack/Thunderbird as a
  stand-alone story (`main.tex` NeurIPS scaffold). This is a *different* paper
  ("SMA as a general structural retriever"); folding it in dilutes the
  golden-ontology thesis. **Recommend: separate paper or drop.**
- **Exploratory rare-disease test** (`scripts/rare_disease_test.py`, the
  "ties top-1, wins top-5/10 vs Phenomizer" curiosity run) — **superseded** by the
  rigorous de-risk A1 + the agentic medicine arm. Cite as "preliminary" at most.
- **Phase 4a LongMemEval drift** — **INVALID** (strawman text-split encoder,
  `docs/STATUS.md` 4a entry). Not reported; at most one line: "free-form
  conversational recall with no ontology is out of scope."
- **Phase 4b dynamic-adapter / hand-built healthcare-expert encoder** experiments
  — exploratory; the *finding* (drafted > hand-expert; functor-vs-entity;
  no-ontology ⇒ parity) informs the boundary discussion, but the specific tabular
  runs are not headline results.
- **Pre-pivot figure set** (`scripts/figures_{main,supp,schematic_tikz}.py`,
  `paper/figures/individual/fig01–13`) — superseded by the new Fig 1 (Claude Design)
  + Fig 2 (matplotlib).

## Open decision for us to agree
The one real fork: **does the Phase 3 SSB/transfer line belong as Extended Data in
this paper (validating the engine), or is it a separate paper?** My recommendation:
**Extended Data** — it cheaply substantiates "the matcher really does match on
structure across a vocabulary gap," which the golden-ontology claim leans on, without
competing for the narrative. Everything in OUT stays out of the results.
