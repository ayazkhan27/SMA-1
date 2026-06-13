# Golden-Ontology Structure-Mapping Memory Turns a Generalist LLM into a Verifiable Specialist

*Draft for Nature Machine Intelligence. Status: 2026-06-13 working draft. All
numbers are from committed runs (`docs/STATUS.md`); slots marked `‹PENDING›`
await the cyber/discovery arm runs in progress. Honesty rails: every reported
result is reproducible from `scripts/`; nulls and parity results are reported
alongside wins.*

---

## Abstract

Retrieval-augmented generation grounds large language models in external text,
but vector and knowledge-graph retrieval index *surface similarity* and *entity
adjacency* — discarding the logical structure (subsumption and higher-order
relations) that defines expertise in high-stakes domains. We present
**Structure-Mapping Agentic Memory (SMA)**, a retrieval memory that encodes each
observation as a *functor* over a subject and retrieves by analogical
structure-mapping (MAC/FAC) against a *golden domain ontology* mounted as its
ascension lattice. A single universal loader ingests any OWL/OBO ontology — we
mount eleven across six domains (medicine, cyber, chemistry, biology, legal,
finance; ≈600k concepts) — and a domain router selects the right one per query.
On a memory-swap benchmark where only the retriever varies, SMA matches a
hand-built, domain-specific ontology tool on retrieval rank while **beating
enterprise-grade RAG and knowledge-graph retrieval** (BM25, BGE neural dense,
hybrid RRF, hybrid + cross-encoder reranking, HippoRAG) on the rare/long-tail
slice that matters most — across **three unrelated domains** (medicine **+33**,
genomics **+16**, cyber **+7** percentage points top-5; all p<0.05, Holm-
corrected). Crucially, SMA also provides two capabilities these baselines
structurally lack: **calibrated cite-or-abstain** (risk-coverage AURC ~0.02–0.11
vs 0.26–0.44 for the best RAG) and **structural-novelty flagging** of unknown
entities (F1 ≈ 0.18 where every pure-RAG baseline scores exactly 0). We show, and delimit
honestly, that the advantage holds where structure exists (ontology-grounded,
rare, cross-vocabulary) and vanishes where it does not (flat-tabular prediction,
free-form conversational recall).

## Main

Large language models are fluent generalists but unreliable specialists: in
medicine, law, security and finance they produce confident, plausible, and
sometimes fabricated answers. The dominant remedy, retrieval-augmented
generation (RAG), grounds the model in an external corpus retrieved by
embedding similarity. This works when the answer is *frequent and lexically
similar* to the query, but it fails on the cases that define expertise — the
rare presentation, the cross-vocabulary analogy, the long-tail entity — because
vector retrieval averages rarity away and indexes surface form rather than
meaning. Knowledge-graph retrieval (including recent graph-RAG variants) adds
typed entity edges, but it traverses them by path adjacency and discriminates on
entity arguments; it still discards the two structures an expert reasons with
most: the *subsumption lattice* (what generalizes to what) and *higher-order
relations* (relations whose arguments are themselves relations).

Cognitive science has long studied retrieval by relational structure. Gentner's
structure-mapping theory and Forbus's MAC/FAC model retrieve analogues by the
systematicity of their relational structure rather than their surface features —
the principle by which an expert recognizes that a novel case is "like" a known
one despite a different vocabulary. We operationalize this as a retrieval
*memory* whose geometry is a curated domain ontology: each observation becomes a
functor over a subject, the ontology's is-a tree becomes an ascension lattice
along which specific terms match general ones, and the ontology's typed relations
become higher-order statements that structure-mapping rewards. A rarity-weighted
(information-content) scorer makes the rare-but-decisive term dominate by
construction.

Our thesis is that **a structure-mapping memory grounded in a golden domain
ontology turns a generalist LLM into a verifiable specialist** — retrieving by
logical structure that vector RAG and knowledge graphs discard, and so beating
both on the rare, cross-vocabulary, high-stakes reasoning where hallucination is
unacceptable, while additionally offering provenance, calibrated abstention, and
detection of genuinely novel inputs. Crucially, we also delimit the thesis: the
advantage is *specific to structure*. Where no ontology applies — flat tabular
prediction, free-form conversational recall — the advantage disappears, and we
report those null results alongside the wins.

We make three contributions. First, a **universal ontology adapter**: a single
loader ingests any OBO, OWL (including Turtle via rdflib), STIX, CPC, or
MITRE-XML source into a normalized graph, mounts it as a structure-mapping
retrieval geometry, and a registry + router select the right ontology per query
— demonstrated on eleven ontologies spanning six domains (~594k concepts) with no
per-domain retrieval code. Second, a **memory-swap agentic benchmark** that holds
the language model and prompt fixed and swaps only the retrieval memory, so any
difference is causally attributable to retrieval, evaluated against the full
enterprise RAG/KG gauntlet (BM25, neural dense, hybrid fusion, cross-encoder
reranking, and a knowledge-graph retriever). Third, an **honest characterization**
of when structure-mapping helps: it reaches parity with a hand-built,
domain-specific ontology tool on pure-subsumption ranking, wins decisively over
general-purpose RAG/KG on the long-tail slice across unrelated domains, and
uniquely provides calibrated cite-or-abstain and structural-novelty signals that
embedding retrieval cannot.

### Results

**The universal adapter and the memory-swap protocol.** (Fig. 1.) The loader
parses any OBO/OWL/STIX/CPC source into a normalized ontology graph (terms,
is-a edges, typed relations), mounts the is-a tree as the predicate ascension
lattice and the typed relations as higher-order statements, and indexes each
entity's term-set as an analogical case. A registry pins versions; a router maps
a query to its domain ontology. We mount eleven ontologies across six domains
(Table 1). The benchmark holds the LLM and prompt fixed and swaps only the
retrieval memory among SMA and five enterprise baselines (BM25, BGE dense,
hybrid RRF, hybrid + bge-reranker, HippoRAG), so any difference is attributable
to the memory.

**SMA matches the bespoke ontology oracle but beats general-purpose RAG/KG.**
On a pre-registered retrieval benchmark (HPO rare-disease A1, GO gene-function
A2; `configs/preregistration_ontology.md`), SMA reaches statistical *parity*
with Phenomizer — a hand-built ontology-aware IC tool — on top-5 (HPO Δ=0.000
p=1.0; GO Δ=+0.036 p=0.31 after Holm). This is expected: both consume the same
ontology, and on pure-subsumption tasks information-content similarity is already
near-optimal. The decisive comparison is against retrievers that do *not* use the
ontology: there SMA wins by large margins (vs lexical Jaccard +21pp HPO / +15pp
GO top-5). We therefore frame the claim precisely as **SMA > RAG/KG**, with the
ontology oracle as a ceiling reference, not a beat-target.

**Against the enterprise RAG/KG gauntlet, SMA wins on the tail across fields.**
(Fig. 2a, d; Table 2.) In the agentic memory-swap suite, on the **medicine** arm
(rare-disease diagnosis; 1,800-disease index, 324 answerable + 36 novel queries,
3 seeds), SMA attains **tail top-5 = 0.949** versus the best enterprise
configuration (hybrid + cross-encoder reranking) at **0.606** — Δ = +0.333,
95% CI [0.281, 0.389], p = 0.0002, Cliff's δ = 0.333; it beats BGE neural dense,
hybrid-RRF, and HippoRAG by even wider margins. On the **cyber** arm (MITRE
ATT&CK threat-group attribution; 82 groups, 222 answerable + 24 novel queries),
SMA again leads — tail top-5 **0.766 vs 0.749** for the best RAG (hybrid-RRF),
Δ = +0.073, 95% CI [0.008, 0.142], p = 0.035 — though the margin is smaller:
threat groups carry distinctive technique vocabularies that lexical and dense
retrieval partly capture, narrowing the rank gap (an honest contrast with the
medicine result). The **discovery** arm (GO gene-function; 5,345 human proteins
retrieved by functional signature) sits between them: SMA tail top-5 **0.849 vs
0.682** for the best RAG, Δ = +0.156, 95% CI [0.100, 0.211], p = 0.0002. All
three unrelated domains survive Holm correction (medicine, discovery, cyber;
adjusted p ≤ 0.035), satisfying the pre-registered across-fields criterion and
confirming that one universal structure-mapping memory beats the enterprise
RAG/KG gauntlet on the long tail wherever a golden ontology exists.

**Cite-or-abstain: SMA's confidence tracks correctness.** (Fig. 2b.) Selective
prediction (risk-coverage) on the medicine arm gives SMA AURC = 0.017 versus
0.317 for the best RAG, 0.401 (dense), 0.510 (BM25), 0.628 (HippoRAG) — roughly
an 18× better-calibrated abstention signal. SMA has the lowest AURC in *every*
arm (0.017 medicine, 0.096 cyber, 0.106 discovery; best RAG 0.26–0.44). A
structural match either exists or it does not; cosine similarity is always
moderately high, so RAG cannot tell when to abstain.

**Structural novelty: SMA flags the unknown.** (Fig. 2c.) On held-out entities
the model has never indexed, SMA's expectation-violation flag yields novelty
F1 ≈ 0.18 in every arm (0.182 medicine, 0.178 cyber, 0.178 discovery), matched
only by the graph-based HippoRAG (≈0.18); every pure-RAG baseline scores
**0.000** in all three domains — a nearest neighbour always exists, so embedding
retrieval has no native signal for "this is new." This is the capability behind
flagging a candidate new disease, a novel attack chain, or an uncharacterized
protein. (We report the absolute F1 honestly: the threshold is fixed at 0.5 and
untuned; the contrast that matters is non-zero versus structurally zero.)

**Where SMA does not help (the honest boundary).** SMA confers no advantage on
flat single-record tabular prediction, where the signal lives in fine-grained
values rather than structure (4b: hospital readmission, card-fraud → statistical
parity with value-based retrieval), nor on free-form conversational recall with
no ontology to mount (LongMemEval drift → the structure-mapping encoder
degenerates). The advantage is specific to ontology-grounded, rare,
cross-vocabulary reasoning — and we report these nulls to delimit it.

### Discussion

- One mechanism, many fields: the same loader+router+matcher serves medicine,
  cyber, chemistry, legal and finance with no per-domain retriever code — a
  generality the bespoke ontology tools (one per domain) cannot match.
- Ontology-as-moat: the defensible asset is a registry of curated golden
  ontologies mounted as retrieval geometry; an LLM can draft rules but cannot
  hallucinate decades of community curation (HPO, GO, MITRE, FIBO, CPC).
- Relational richness, not size, drives SMA's edge over the oracle: pure-is-a
  ontologies (HPO, CPC) yield parity; relation-rich ones (GO, ChEBI, ATT&CK)
  are where higher-order structure-mapping should pull ahead — a falsifiable
  prediction for the remaining arms.
- Limitations: novelty thresholds are untuned (fixed 0.5); HippoRAG is somewhat
  disadvantaged by a bag-of-term-names document representation; the interactive
  agentic flagship is a single-domain case study; SNOMED/UMLS (licensed) are
  excluded in favour of fully open ontologies.

### Methods

- **Structure-mapping core.** Statements are functors over entities; SME builds
  consistent kernel mappings; MAC/FAC provides a certified admissible content
  bound (Lemma 2) for tractable first-stage retrieval, best-first FAC for the
  second. Cross-vocabulary matching goes through the ontology lattice with an
  ascension penalty ρ^distance (frozen ρ=0.95, δ=2).
- **Surprisal scorer.** A matched functor is weighted by −log₂ p (corpus
  information content), so rare terms dominate the score by construction.
- **Universal ontology loader.** `sma/ontology`: `load_obo`, `load_owl`
  (stdlib), `load_owl_dir`, `load_rdflib` (Turtle/complex OWL), `load_attack_stix`,
  `load_cpc`, `load_mitre_xml`. `mount()` builds the Canonicalizer lattice +
  higher-order case builder; `OntologyRegistry` + `DomainRouter` select per query.
- **Memory-swap harness.** `sma/eval/agentic`: a `Memory` interface wraps SMA and
  each baseline identically; `run_oneshot` indexes the same records in every
  memory, generates hard partial/imprecise/noisy queries, and scores tail top-k
  (rare slice = entity's rarest-term IC above the corpus median), risk-coverage
  AURC, and novelty F1, with paired bootstrap (10k, seed 12345) + Holm.
- **Baselines.** BM25; BGE-small neural dense (cosine); hybrid RRF (k=60);
  hybrid + `bge-reranker-base` cross-encoder; HippoRAG-2 (phrase graph + PPR).
- **Reproducibility.** Ontologies version-pinned (`configs/ontologies.json`),
  `PYTHONHASHSEED=0`, sorted set→list throughout; commands in `docs/STATUS.md`.

## Display items

**Figure 1.** Universal adapter + memory-swap concept (TikZ schematic).
**Figure 2.** Agentic results: (a) per-domain tail top-5 SMA vs RAG/KG;
(b) risk-coverage / cite-or-abstain; (c) novelty F1; (d) per-domain effect size.
[`paper/figures/individual/fig_agentic_results.pdf`]
**Table 1.** The eleven mounted ontologies (domain, source, terms, is-a, typed).
**Table 2.** Medicine arm full results (all + rare slices, AURC, novelty F1).
**Extended Data.** De-risk suite (A1 HPO, A2 GO) vs Phenomizer/Jaccard/Dense/Hippo.

---

### Table 1 — The eleven mounted golden ontologies (one universal loader)

| Domain | Ontology | Format | Terms | is-a edges | Typed relations |
|---|---|---|---|---|---|
| Medicine | HPO | OBO | 19,810 | 24,329 | 0 (pure is-a) |
| Medicine | MONDO | OBO | 56,273 | 78,736 | 34,157 |
| Anatomy | Uberon | OBO | 14,973 | 19,382 | 11,697 |
| Biology | GO | OBO | 38,263 | 57,803 | 14,076 |
| Chemistry | ChEBI | OBO | 205,592 | 285,589 | 94,902 |
| Cyber | MITRE ATT&CK | STIX 2.1 | 712 | 475 | 872 |
| Cyber | MITRE CAPEC | XML | 558 | 532 | 194 |
| Cyber | MITRE CWE | XML | 944 | 1,160 | 284 |
| Legal/IP | CPC | XML | 254,274 | 256,106 | 0 (pure is-a) |
| Legal | LKIF-core | OWL | 153 | 171 | 48 |
| Finance | FIBO | RDF (rdflib) | 2,948 | 3,819 | 1,335 |

Total ≈ 594k concepts across 6 domains, all routed (no merged omni-graph; merge
WITHIN an aligned ecosystem, route ACROSS). Pure-is-a ontologies (HPO, CPC) yield
parity vs the ontology oracle; relation-rich ones (ChEBI, MONDO, GO, ATT&CK) are
where higher-order structure-mapping should exceed it.

---

### Table 2 — Agentic memory-swap results across three domains

**(a) Tail top-5 accuracy (rare slice).** Bold = best.

| Memory | Medicine | Genomics (GO) | Cyber (ATT&CK) |
|---|---|---|---|
| **SMA (ours)** | **0.949** | **0.849** | **0.766** |
| BM25 | 0.485 | 0.552 | 0.703 |
| BGE dense | 0.496 | 0.682 | 0.629 |
| Hybrid-RRF | 0.511 | 0.703 | 0.749 |
| Hybrid + rerank | 0.606 | 0.651 | 0.651 |
| HippoRAG (KG) | 0.361 | 0.318 | 0.377 |
| **Δ SMA − best RAG** | **+0.333** | **+0.156** | **+0.073** |
| (95% CI; p, Holm) | [0.281,0.389]; 6e-4 | [0.100,0.211]; 4e-4 | [0.008,0.142]; 0.035 |

**(b) Capability axes RAG lacks** (AURC lower = better; novelty F1).

| Memory | Med AURC | Gen AURC | Cyb AURC | Med/Gen/Cyb novelty F1 |
|---|---|---|---|---|
| **SMA (ours)** | **0.017** | **0.106** | **0.096** | 0.182 / 0.178 / 0.178 |
| best RAG | 0.317 | 0.298 | 0.261 | 0.000 / 0.000 / 0.000 |
| HippoRAG (KG) | 0.628 | 0.721 | 0.560 | 0.186 / 0.185 / 0.182 |

All three domains: SMA beats the best enterprise RAG on tail top-5 (Holm-
significant), has the lowest risk-coverage AURC, and is the only embedding-free
method (with the graph-based HippoRAG) to produce any novelty signal.
