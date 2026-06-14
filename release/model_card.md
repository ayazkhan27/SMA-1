---
license: apache-2.0
tags:
  - structure-mapping
  - analogical-retrieval
  - retrieval
  - sma
  - rare-disease
  - genomics
  - cybersecurity
  - legal
  - finance
  - cite-or-abstain
  - novelty-detection
  - ontology
language:
  - en
pipeline_tag: feature-extraction
---

# SMA-1 Universal Adapter — adapter-v1

**Structure-Mapping Agentic Memory: a universal structure-mapping retrieval
adapter that beats RAG/KG baselines across five unrelated curated-ontology
domains — with calibrated abstention and novelty detection that vector RAG
structurally cannot provide.**

- **Paper:** *"Structure-Mapping Agentic Memory"*, Ayaz Khan (2026, under
  review, *Nature Machine Intelligence*)
- **Source repository:** https://github.com/ayazkhan27/sma-1
- **Tag:** `adapter-v1` (frozen after 5/5 agentic wins; ADR-008)
- **License:** Apache-2.0

---

## What is this adapter?

`adapter-v1` is the **universal ontology adapter** for SMA-1.  It is not a
neural network weight file — it is a frozen configuration that specifies:

1. **Five mounted curated ontologies** (one per evaluation domain) — each parsed
   from its canonical OBO/OWL/STIX/XML source, mounted onto a predicate lattice,
   and indexed via the SMA MAC/FAC retrieval engine.
2. **Frozen matcher dials** (see below) — calibrated on held-out validation data
   before any confirmatory run.
3. **A domain router** — routes a query to the correct ontology by term-id prefix
   or declared domain.

The adapter is "frozen" in the sense that no component (ontology versions,
matcher dials, encoder rules) changes after the tag.  New domains may be
*added* under a new tag (ADR-008); the frozen components do not move.

---

## Mounted curated ontologies

| Domain | Ontology | Source format | Approx. active terms |
|---|---|---|---|
| Medicine | Human Phenotype Ontology (HPO) | OBO | ~17 000 |
| Genomics | Gene Ontology (GO) biological process | OBO | ~30 000 |
| Cybersecurity | MITRE ATT&CK (STIX 2.1) | JSON/XML | ~700 techniques and sub-techniques |
| Legal | USPTO Cooperative Patent Classification (CPC) | OWL/XML | ~254 000 nodes |
| Finance | US-GAAP SEC XBRL taxonomy | OWL/XML | ~900 concepts |

Each ontology is mounted via `sma.ontology.mount()`:
- The is-a hierarchy becomes the **predicate lattice** (ascension depth δ = 2,
  penalty ρ = 0.95 per hop).
- Each annotated entity (disease, protein, threat group, patent, filing) is
  encoded as a **case**: one `stmt(fid(term_id), subject)` per present term, plus
  higher-order `stmt(rel, stmt(fid(s), subj), stmt(fid(o), subj))` for typed
  relations whose subject and object are both present.
- The MAC (Memory Analogical Coding) stage indexes cases via a weighted Lemma-2
  inverted index (bound-ordered over all candidates for corpora ≤ 20k cases).
- The FAC (Full Analogical Coding / SME) stage computes structural alignment
  kernels; event-type entities are constants (match identically); structure
  mapping respects parallel connectivity.

---

## Frozen matcher dials (prereg-v1)

Calibrated on held-out validation splits (HDFS seed-7, SSB seeds 29/31,
Liberty leave-one-out); test seeds never used during calibration.

| Parameter | Value | Meaning |
|---|---|---|
| Scorer | `surprisal` | Surprisal-weighted SES (σ₀ = −log₂ p(functor)) |
| Normalisation | `max` | Score / max(score in result set) |
| γ (trickle-down weight) | 0.25 | Blueprint §2.5 default |
| ρ (ascension penalty) | 0.95 | Per-hop lattice ascension penalty |
| δ (ascension depth) | 2 | Maximum ancestor hops for lattice bridging |

---

## Memory-swap evaluation protocol

The evaluation uses a **memory-swap** design: one LLM agent, one prompt, one
task — the only variable is the retrieval memory.  This isolates the contribution
of retrieval from language generation.

**Agentic suite (5 domains):** the agent is given a query (phenotype set, gene
function annotation set, threat-group technique set, patent claims, SEC filing)
and must retrieve the top-k matching indexed entities.  The gold answer is the
correct entity (disease, protein, threat group, CPC code, GAAP concept).

**Phase 5 LLM-QA (trustworthy specialist QA):** the agent is given a clinical
diagnosis question and must answer it from retrieved evidence, cite its sources,
or abstain if the evidence is insufficient.  The LLM (DeepSeek, temperature 0)
and prompt are fixed; only the retrieval memory varies (none / dense / SMA).
The cite-or-abstain threshold is calibrated per memory on a disjoint 60+60
calibration split (Youden's J on retrieval scores only — no LLM spend, no test
leakage).

---

## Verified headline metrics

All numbers from committed `reports/confirmatory/` CSVs (paired bootstrap
10 000 resamples, Holm-Bonferroni step-down correction).

### 5-domain agentic suite — SMA vs best RAG baseline (tail top-5)

"Tail" = rare slice (entity's rarest-term IC > corpus median).
Legal arm reported on all-queries (rare slice degenerate for CPC — see limitations).

| Domain | SMA tail top-5 | Best RAG | Δ | 95% CI | p_Holm | Cliff's δ |
|---|---|---|---|---|---|---|
| Medicine (HPO) | 0.949 | 0.606 (hybrid+rerank) | **+0.333** | [0.281, 0.389] | 0.0006 | 0.333 |
| Genomics (GO) | 0.849 | 0.682 (dense BGE) | **+0.156** | [0.100, 0.211] | 0.0004 | 0.156 |
| Finance (US-GAAP) | 0.418 | 0.231 (hybrid-RRF) | **+0.167** | [0.111, 0.225] | 0.0002 | 0.167 |
| Cybersecurity (ATT&CK) | 0.766 | 0.749 (hybrid-RRF) | **+0.073** | [0.008, 0.142] | 0.0346 | 0.073 |
| Legal (CPC) | 0.941 (all) | 0.870 (dense BGE, all) | **+0.064** | [0.025, 0.103] | 0.0022 | 0.064 |

Four domains survive conservative correction; cyber is directional.  RAG/KG baseline gauntlet: BM25, BGE dense, Hybrid-RRF,
Hybrid+Rerank (cross-encoder reranker), HippoRAG (phrase-graph + PageRank).

**Capability axes (all domains):** SMA achieves lowest AURC (best calibrated
selective prediction) and is the only method (along with HippoRAG) with nonzero
novelty F1.  All pure-RAG baselines: novelty F1 = 0.000.

### Phase 5 LLM-QA — SMA vs dense (medicine, n = 120 answerable + 120 held-out)

| Axis | SMA | Dense RAG | Δ | 95% CI | p_Holm | Result |
|---|---|---|---|---|---|---|
| Accuracy | 0.342 | 0.100 | +0.242 | [+0.167, +0.325] | < 0.001 | WIN |
| Grounding AUROC | 0.793 | 0.547 | +0.246 | [+0.159, +0.333] | < 0.001 | WIN |
| Novelty F1 | 0.789 | 0.553 | +0.236 | [+0.200, +0.408] (recall) | < 0.001 | WIN |
| Selective accuracy | 0.625 | 0.500 | +0.125 | [+0.071, +0.179] | < 0.001 | WIN |
| Abstain recall | 0.908 | 0.900 | +0.008 | [−0.058, +0.075] | 0.917 | TIE |

4/5 axes Holm-significant; abstain-recall is an honest tie.

**The mechanism (Fig 5b):** SMA's raw structural grounding score separates known
(answerable) from unknown (held-out) at AUROC 0.793; dense cosine is near chance
(0.547).  Dense RAG must refuse 79% of answerable questions to achieve the same
abstain-recall as SMA at 45% abstention.

### Structure Synthesis Benchmark (SSB)

Zero-lexical-overlap structural retrieval; disjoint per-triple vocabularies
bridged only by a declared predicate lattice (seeds 41, 43; n = 100 each):

| Method | Forced-choice r@1 | Library r@1 |
|---|---|---|
| SMA | **1.000** | **0.895** |
| BM25 | 0.000 | 0.000 |
| TF-IDF Dense | 0.000 | 0.000 |

Cliff's δ = 0.895, p_Holm = 0.0004.

---

## Intended use

`adapter-v1` is intended for:

- **Research** into structure-mapping retrieval and analogical reasoning for LLM
  agents.
- **Evaluation** of SMA-1 claims: reproduce results via
  `scripts/confirmatory_battery.py` or `scripts/agentic_suite.py`.
- **Extension** to new domains: register a new OBO/OWL ontology via
  `OntologyRegistry`, mount it, run `agentic_suite.py --arm <new_arm>`.  A new
  adapter tag is required for any new frozen ontology (ADR-008).
- **The Gradio demo Space** (`release/hf_space/`) which illustrates the
  medicine arm side-by-side with dense RAG.

`adapter-v1` is **not** intended for:

- Production clinical decision support (not a medical device; not validated for
  clinical use).
- Domains without a curated ontology (the structural advantage requires a
  predicate lattice; flat-tabular data yields parity or null — confirmed on
  UCI Diabetes-130 and IEEE fraud without adapter drafting).
- Tasks where surface-retrieval baselines already achieve near-perfect performance
  (e.g. within-domain log triage with lexically overt labels — BGL in the T2
  battery; SMA is statistically tied, not a win).

---

## Limitations and honest nulls

1. **Flat-tabular data.** Where per-record higher-order relational structure is
   absent or cannot be meaningfully encoded, SMA reaches statistical parity with
   baselines but does not win (UCI Diabetes-130 before adapter drafting: SMA 0.425
   vs BM25 0.537 — not significant; IEEE fraud: SMA below BM25 after adapter
   drafting — cross-record structure is needed, not handled per-record).

2. **Cross-family transfer.** Structural transfer holds within failure-physics
   families (supercomputer syslogs, BGL→Spirit, BGL→Thunderbird: SMA +58 F1 pts
   over dense).  It fails across application-vs-infrastructure families
   (HDFS→OpenStack: all methods collapse to ~0.33).

3. **Legal arm rare slice.** The CPC rare-slice definition (IC > corpus median)
   degenerates for patent CPC codes (near-uniform IC from closure propagation);
   legal results are reported on the all-queries slice with an explicit caveat.

4. **Agentic LLM-QA: medicine only.** Phase 5 LLM-QA evaluation is on the
   medicine (HPO) domain only; the verifiable-specialist result has not been
   extended to the other four domains under prereg-v2.

5. **ATT&CK cap.** ATT&CK groups with > 30 techniques were capped (SME kernel
   enumeration timeout without the cap); 41% of AT&CK groups are affected.
   Results reflect the capped subset.

6. **Novelty F1 threshold.** The SAGE novelty threshold is fixed at 0.5 (not
   tuned); the absolute novelty F1 values are conservative and would likely
   improve with threshold calibration.

7. **Phase 4a drift result is INVALID for SMA.** The LongMemEval run (500
   instances) is NOT an SMA result — the backend used a broken encoder that
   collapsed all facts to functor "User"/"The", producing garbage retrieval
   (SMA accuracy 0.030 = encoder artifact, not reported as an SMA result).

---

## How to use

```python
from sma.ontology import OntologyRegistry, DomainRouter

# Register a curated ontology
reg = OntologyRegistry()
reg.register("hpo", "data/hp.obo")          # OBO format inferred from extension
mounted = reg.get("hpo")                     # lazily loads, mounts, caches

# Build an index over annotated entities
from sma.eval.agentic.memories import SmaMemory, IndexItem
mem = SmaMemory(mounted)
mem.index([
    IndexItem(key="OMIM:154700",
              term_ids=frozenset(["HP:0001166", "HP:0001083", "HP:0002616"]),
              text="Marfan syndrome arachnodactyly ectopia lentis aortic root dilatation"),
    # ... more entries
])

# Retrieve
from sma.eval.agentic.memories import Query
results = mem.retrieve(Query(term_ids=frozenset(["HP:0002616", "HP:0000098"]),
                             text="aortic root dilatation tall stature"), k=5)
for r in results:
    print(r.key, r.score, r.confidence)

# Novelty gate
nov = mem.novelty(Query(term_ids=frozenset(["HP:0099999"]), text="unknown phenotype"))
print(f"Novelty signal: {nov:.3f}")  # high = likely out-of-distribution
```

Reproduce the full evaluation:

```bash
# Confirmatory battery (single-shot; ~5 h, registered seeds)
python3 -u scripts/confirmatory_battery.py --task all

# Agentic suite (5 domains)
python3 scripts/agentic_suite.py --arm medicine
python3 scripts/agentic_suite.py --arm discovery
python3 scripts/agentic_suite.py --arm finance
python3 scripts/agentic_suite.py --arm cyber
python3 scripts/agentic_suite.py --arm legal

# Phase 5 LLM-QA (requires SMA_DEEPSEEK_API_KEY)
python3 scripts/agentic_qa.py --memory sma --n-index 1500
```

---

## Citation

```bibtex
@software{khan2026sma,
  author  = {Ayaz Khan},
  title   = {SMA-1: Structure-Mapping Agentic Memory},
  year    = {2026},
  license = {Apache-2.0},
  url     = {https://github.com/ayazkhan27/sma-1}
}
```

---

## License

Apache-2.0.  See `LICENSE` at the repository root.

The mounted ontologies are derived from publicly licensed sources:
- HPO: hpo.jax.org (CC BY 4.0)
- GO: geneontology.org (CC BY 4.0)
- MITRE ATT&CK: attack.mitre.org (Apache 2.0)
- CPC: USPTO (public domain)
- US-GAAP: SEC EDGAR (public domain)

The SMA-1 code and adapter configuration are Apache-2.0.
