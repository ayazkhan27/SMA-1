# 5-Domain Agentic Ontology Benchmark Suite — Design

Status: **approved 2026-06-13** (brainstorming). Companion to `docs/PAPER_SPINE.md`
and `configs/preregistration_ontology.md`. This spec covers the **harness + the
Medicine arm** (one-shot core + interactive flagship). The other four domains are
arm-instances that reuse the harness; each gets a short follow-up spec when its
benchmark data lands. All five ontologies are loaded into the registry for
domain routing regardless of arm-build order.

## 1. Goal

Prove the paper spine against *real* opponents: a generalist LLM, equipped with
the universal-adapter SMA memory grounded in a golden ontology, beats
**enterprise-grade RAG and KG retrieval** on the rare/tail slice — and adds
verifiable capabilities (cite-or-abstain, structural novelty) that those
baselines structurally cannot provide. One mechanism, five fields.

## 2. Why (anchored in today's de-risk result)

SMA *ties* the bespoke ontology oracle (Phenomizer) on rank but *beats* TF-IDF
RAG by +29pp and HippoRAG by +60pp (HPO, tail). The headline is therefore
**tail accuracy vs RAG/KG**, which only counts if the RAG is state-of-the-art —
hence the enterprise baseline gauntlet (§5). Hypothesis to test across domains:
**SMA's edge over IC-similarity grows with higher-order relational density**
(HPO/GO are near-pure is-a → tie; ATT&CK/legal/finance have rich typed relations
→ SMA should pull ahead).

## 3. Architecture — one harness, two layers

```
sma/eval/agentic/
  harness.py     # memory-swap A/B core; fixed LLM+prompt, swap only the retriever
  memories.py    # uniform Memory interface wrapping each retriever (incl. SMA)
  metrics.py     # tail accuracy, risk-coverage (selective prediction), novelty F1
  interactive.py # AgentClinic-style loop (flagship only), budget-capped
  arms/
    medicine.py  # the first arm: HPO/MONDO/GO/Uberon + gold loader + query gen
```

- **One-shot core (all domains):** the agent receives a query, calls ONE memory
  backend (the swapped variable), retrieves top-k grounded candidates, and emits
  `{answer, citation, abstain, novelty_flag}`. Everything except the memory is
  held fixed → clean causal attribution.
- **Interactive flagship (Medicine only):** an ask→retrieve→converge loop; the
  agent iteratively requests evidence, queries memory each step, converges to a
  ranked differential. Budget-capped (max N turns, logged token spend).

### 3.1 Memory interface (the only thing that varies)

```python
class Memory(Protocol):
    name: str
    def index(self, records: Iterable[tuple[str, set[str], dict]]) -> None: ...
    def retrieve(self, query_terms: set[str], k: int) -> list[Retrieved]: ...
    # Retrieved: {key, score, evidence_terms, confidence}  (confidence drives abstain)
```

Wraps: SMA (universal adapter), and each baseline in §5. The harness never sees
retriever internals — only this interface — so adding a baseline or a domain is a
new object, not a new code path.

## 4. Metrics (pre-registered)

- **Headline — tail retrieval accuracy.** top-k (k∈{1,5,10}) on the rare slice
  (entity's rarest term IC > corpus median), SMA vs the *best enterprise-RAG*
  config. **Falsifier:** paired-bootstrap (10k, seed 12345) 95% CI for
  Δtop-5(SMA − best-RAG) includes 0 after Holm across arms.
- **Secondary — cite-or-abstain (selective prediction).** Risk–coverage curve +
  AURC: as the agent abstains on low-confidence queries, does error drop faster
  for SMA than RAG? Built by withholding answerable vs unanswerable (out-of-KB)
  queries and sweeping the confidence threshold.
- **Secondary — structural-novelty F1.** Hold out a set of entities entirely;
  their queries are "novel." SMA flags via `SagePool.expectation_violation`;
  RAG/KG have no native signal (use max-similarity threshold as their best
  proxy). Report F1(novel vs known) per method.
- **Supporting — generality.** One harness, all domains, no per-domain retriever
  code: reported as a fact + a per-domain results table.

Reuses `sma/eval/stats.py` (paired_bootstrap, holm_bonferroni, cliffs_delta).

## 5. Baseline gauntlet (per arm) — enterprise-grade RAG

| # | Baseline | Implementation |
|---|---|---|
| 1 | BM25 | `sma/eval/baselines/bm25.py` (have) |
| 2 | Neural dense | **BGE/E5 sentence-transformers** over term-name docs → cosine (NEW, local) |
| 3 | Hybrid (BM25+dense, RRF) | reuse Phase-3 Hybrid-RRF, dense = BGE |
| 4 | Hybrid + cross-encoder rerank | reuse Phase-3 Hybrid+Rerank, reranker = `bge-reranker` |
| 5 | HippoRAG-2 (KG) | `sma/eval/baselines/hipporag.py` (have) |
| 6 | **SMA** (universal adapter) | `sma/ontology` mount + MacFacIndex |
| – | Phenomizer / IC oracle | ceiling reference, NOT a beat-target |

Embedding backend: **local open SOTA** (`sentence-transformers`, BGE/E5 +
bge-reranker). Requires installing `sentence-transformers` (+torch) — sanctioned;
exact packages flagged before install. CPU-runnable on short term-name docs.

## 6. Domains (5) — registry + routing + build order

All five ontologies registered in `configs/ontologies.json` for routing
(prefix→ontology). Arm build order by data-readiness:

| # | Domain | Ontology | Gold benchmark | Build |
|---|---|---|---|---|
| 1 | **Medicine** (flagship) | HPO+MONDO+GO+Uberon | phenotype.hpoa (+RareBench-style) | **this spec** |
| 2 | Cyber | MITRE ATT&CK | TRAM / CTI→technique | follow-up |
| 3 | Legal | LKIF / CPC / caselaw | LegalBench / COLIEE | follow-up |
| 4 | Finance | FIBO | AML typology matching | follow-up |
| 5 | Discovery | ChEBI + GO + MeSH | drug-repurposing / LBD | follow-up |

## 7. The Medicine arm (built now)

- **Records:** diseases ← `phenotype.hpoa` (aspect P, 7–30 phenotypes), mounted
  via `sma.ontology` (HPO is-a lattice). Reuses the proven `ontology_bench` query
  generator (hard partial/imprecise patients).
- **One-shot:** all six retrievers indexed over the same disease records; agent
  answers/abstains/flags per query; metrics §4.
- **Interactive flagship:** AgentClinic-style differential-diagnosis loop on a
  small held-out case set, DeepSeek-backed, budget-logged.
- **Regression anchor:** the one-shot SMA-vs-RAG/KG numbers must match the
  `ontology_suite_v2` de-risk run (SMA ≫ Dense/HippoRAG; ≈ oracle).

## 8. Data flow

`load_obo/load_attack_stix → mount → MountedOntology` → arm loader builds
`records` → harness indexes every Memory → query generator emits hard queries →
each Memory retrieves → agent (one-shot or interactive) emits decision → metrics
→ paired-bootstrap + Holm → CSV + STATUS entry.

## 9. Error handling & honesty rails

- Every baseline that under-indexes (e.g., empty embedding) logs a warning, never
  silently scores 0.
- Report the FULL curve (all slice + rare slice), not just the tail, per the
  anti-cherry-pick rule.
- If SOTA neural RAG erases the tail win, report it; the capability metrics
  (abstain/novelty) carry the differentiated claim.
- Budget guard on the interactive loop: hard token cap, logged spend, abort-safe.

## 10. Testing

- `tests/agentic/test_harness.py` — memory-swap isolation (same query → each
  Memory returns its own ranking; harness identical otherwise).
- `tests/agentic/test_metrics.py` — risk-coverage + novelty-F1 on synthetic
  fixtures with known answers.
- `tests/agentic/test_medicine_arm.py` — small-N regression vs the de-risk numbers.
- Existing `tests/ontology/` stays green.

## 11. Out of scope (this spec)

The four follow-up arms' gold-task construction; the real RareBench corpus
licensing; any hosted-API embedding. These are separate short specs.
