# Agentic Ontology Suite (Harness + Medicine Arm) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or a Workflow to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build a memory-swap agentic benchmark harness that proves SMA (universal ontology adapter) beats enterprise RAG/KG on the rare/tail slice, with cite-or-abstain + structural-novelty metrics, validated on the Medicine arm.

**Architecture:** One harness; the *only* variable is the retrieval `Memory` (SMA vs BM25 vs BGE-dense vs Hybrid-RRF vs Hybrid+Rerank vs HippoRAG). Each domain is a thin arm `(mounted ontology, entity→term-set records, query generator, novel-entity holdout)`. Metrics: tail top-k (headline), risk-coverage AURC (cite-or-abstain), novelty F1.

**Tech Stack:** Python, `sma.ontology` (mount/MacFacIndex), `sma.eval.baselines.{bm25,dense,hipporag}`, `sentence-transformers` (BGE embed + bge-reranker, CPU), `sma.eval.stats` (paired bootstrap, Holm).

---

## File Structure

- `sma/eval/agentic/__init__.py` — exports.
- `sma/eval/agentic/memories.py` — `IndexItem`, `Query`, `Retrieved`, `Memory` protocol + 6 wrappers.
- `sma/eval/agentic/metrics.py` — `tail_topk`, `risk_coverage_aurc`, `novelty_f1`.
- `sma/eval/agentic/harness.py` — `run_oneshot(arm, memories, ...)` orchestration.
- `sma/eval/agentic/arms/medicine.py` — Medicine arm (HPO records + query gen + holdout).
- `scripts/agentic_suite.py` — driver (one-shot core).
- `tests/agentic/{test_memories,test_metrics,test_harness,test_medicine_arm}.py`.

Interactive flagship (`interactive.py`) is a SEPARATE follow-up task after the one-shot core is green (it needs DeepSeek + budget guard).

---

## Task 1: Memory interface + data types

**Files:** Create `sma/eval/agentic/memories.py`; Test `tests/agentic/test_memories.py`

- [ ] **Step 1: dataclasses + protocol**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, Iterable

@dataclass
class IndexItem:
    key: str                 # entity id (gold answer)
    term_ids: frozenset[str] # ontology term ids (for SMA)
    text: str                # space-joined term NAMES (for text baselines)
    meta: dict = field(default_factory=dict)

@dataclass
class Query:
    term_ids: frozenset[str]
    text: str

@dataclass
class Retrieved:
    key: str
    score: float
    confidence: float        # drives cite-or-abstain; method-specific, in [0,1]
    rank: int

class Memory(Protocol):
    name: str
    def index(self, items: list[IndexItem]) -> None: ...
    def retrieve(self, query: Query, k: int) -> list[Retrieved]: ...
    def novelty(self, query: Query) -> float: ...   # higher = more "this is new"
```

- [ ] **Step 2: test the contract on a stub** — assert a trivial `Memory` returns `Retrieved` with monotonic ranks. Commit.

## Task 2: SMA memory (universal adapter)

**Files:** Modify `sma/eval/agentic/memories.py`; Test `tests/agentic/test_memories.py`

- [ ] SMA wrapper. `index`: build cases via `mounted.build_case(item.term_ids)`, build `MacFacIndex(config=mounted.config, canon=mounted.canon)`, store `key_of`. Also build a `SagePool("agentic")` and `assimilate` each case for novelty. `retrieve`: `mounted.build_case(query.term_ids)` → `index.retrieve(k, shortlist=80, fac_budget=40)`; `confidence` = top result's normalized score (clip [0,1]); `rank` 1..k. `novelty`: `pool.expectation_violation(query_case)` (already [0,1]).

```python
from sma.ontology import MountedOntology
from sma.index.macfac import MacFacIndex
from sma.sage.pools import SagePool

class SmaMemory:
    name = "sma"
    def __init__(self, mounted: MountedOntology):
        self.mounted = mounted
    def index(self, items):
        self._key = {}
        cases = []
        self.pool = SagePool("agentic", assimilation_threshold=0.2)
        for it in items:
            c = self.mounted.build_case(it.term_ids, metadata={"key": it.key})
            self._key[c.case_id] = it.key
            cases.append(c); self.pool.assimilate(c)
        self.index_ = MacFacIndex(config=self.mounted.config, canon=self.mounted.canon)
        self.index_.build(cases)
    def retrieve(self, query, k):
        qc = self.mounted.build_case(query.term_ids)
        res = self.index_.retrieve(qc, k=k, shortlist=80, fac_budget=40)
        top = res[0].score if res else 0.0
        return [Retrieved(self._key.get(r.case_id, ""), r.score,
                          min(max(top, 0.0), 1.0), i) for i, r in enumerate(res, 1)]
    def novelty(self, query):
        return self.mounted_pool_violation(query)
    def mounted_pool_violation(self, query):
        return self.pool.expectation_violation(self.mounted.build_case(query.term_ids))
```

- [ ] Test: index 3 toy entities, retrieve an exact-shape query → true key rank 1, confidence>0; a query of unrelated terms → higher novelty than an in-distribution query. Commit.

> NOTE for implementer: confirm `RetrievalResult` has `.score` and `.case_id` (it does). If `.score` is unnormalized, divide by the max over results to keep confidence in [0,1].

## Task 3: Text baselines — BM25, Dense (BGE), HippoRAG

**Files:** Modify `sma/eval/agentic/memories.py`; Test `tests/agentic/test_memories.py`

- [ ] BM25Memory: wrap `sma.eval.baselines.bm25.rank_bm25_like` over `[(key,text)]`; confidence = top score / (top+1) (squash to [0,1]); novelty = 1 - that confidence.
- [ ] DenseMemory: `sentence_transformers.SentenceTransformer("BAAI/bge-small-en-v1.5")`. `index`: encode all texts (normalize_embeddings=True), store matrix + keys. `retrieve`: encode query.text, cosine vs matrix, top-k; confidence = top cosine (already [0,1]); novelty = 1 - top cosine. Cache the model at module level so it loads once.
- [ ] HippoMemory: wrap `sma.eval.baselines.hipporag.HippoRAGRetriever` (build once on index, retrieve per query); confidence = top score normalized by sum; novelty = 1 - confidence.

```python
from sma.eval.baselines.bm25 import rank_bm25_like
from sma.eval.baselines.hipporag import HippoRAGRetriever
_BGE = {}
def _bge():
    if "m" not in _BGE:
        from sentence_transformers import SentenceTransformer
        _BGE["m"] = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _BGE["m"]
```

- [ ] Test: each baseline indexes 3 toy docs and ranks an exact-match query's key at 1; confidence in [0,1]. (Mark the BGE test `@pytest.mark.slow` and skip if no network/model.) Commit.

## Task 4: SOTA hybrid — Hybrid-RRF + Hybrid+Rerank

**Files:** Modify `sma/eval/agentic/memories.py`; Test `tests/agentic/test_memories.py`

- [ ] HybridRRFMemory(bm25_mem, dense_mem, k_rrf=60): retrieve from both, fuse by Reciprocal Rank Fusion `score = sum 1/(k_rrf + rank)`; confidence = normalized top fused score; novelty = min of the two members' novelty.
- [ ] HybridRerankMemory(hybrid, cross_encoder="BAAI/bge-reranker-base", top_n=30): take hybrid top_n, rerank with `sentence_transformers.CrossEncoder` on `(query.text, item.text)`; confidence = sigmoid(top reranker logit); needs an id→text map captured at index time.

- [ ] Test: RRF of two toy member memories ranks the doc both agree on first; rerank reorders by cross-encoder score. (`@pytest.mark.slow` for the reranker.) Commit.

## Task 5: Metrics

**Files:** Create `sma/eval/agentic/metrics.py`; Test `tests/agentic/test_metrics.py`

- [ ] `tail_topk(rows, k)` — `rows`: list of `{method: rank, rare: bool}`; returns per-method top-k on all + rare slice (reuse the convention from `sma/eval/ontology_bench.py`).
- [ ] `risk_coverage_aurc(confidences, correct)` — sort by confidence desc; sweep coverage 0→1; risk = error rate over the covered head; return AURC (lower is better) + the curve points.

```python
def risk_coverage_aurc(confidences, correct):
    order = sorted(range(len(correct)), key=lambda i: -confidences[i])
    n = len(correct); cum_err = 0; pts = []
    for j, i in enumerate(order, 1):
        cum_err += (0 if correct[i] else 1)
        pts.append((j / n, cum_err / j))           # (coverage, risk)
    aurc = sum(r for _, r in pts) / max(len(pts), 1)
    return aurc, pts
```

- [ ] `novelty_f1(novel_flags_pred, novel_truth)` — standard F1 of predicted-novel vs truly-held-out.

```python
def novelty_f1(pred, truth):
    tp = sum(1 for p, t in zip(pred, truth) if p and t)
    fp = sum(1 for p, t in zip(pred, truth) if p and not t)
    fn = sum(1 for p, t in zip(pred, truth) if not p and t)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0
```

- [ ] Tests: AURC of a perfectly-calibrated ranker (all correct first) < AURC of random; novelty_f1 on a known split. Commit.

## Task 6: Harness

**Files:** Create `sma/eval/agentic/harness.py`; Test `tests/agentic/test_harness.py`

- [ ] `run_oneshot(name, mounted, records, memories, *, seeds, n_index, n_query, holdout_frac=0.1)`:
  - `records`: `dict[key -> set[term_id]]`. Split a `holdout_frac` of entities as NOVEL (their queries are unanswerable → used for novelty + abstain). Index only the non-holdout entities in every memory.
  - Build hard queries (reuse `ontology_bench` generator: sample 5 terms, climb 0–2, +3 noise) for both in-distribution (answerable) and holdout (novel) entities. Compute `text` from term names.
  - For each memory: `retrieve` → rank of true key (999 if novel/absent); record `confidence`, `novelty(query)`, `rare` flag, `is_novel` flag.
  - Aggregate per memory: tail top-k (answerable, all+rare), risk-coverage AURC (answerable correct vs confidence), novelty F1 (novelty>thr vs is_novel; thr chosen per method to max F1 on a held val split — or fixed 0.5 with a noted caveat).
  - Stats: SMA vs best enterprise-RAG (best of bm25/dense/hybrid/rerank/hippo) on tail top-5 via `paired_bootstrap`; Holm later across arms.
  - Return a result dict; the harness must be deterministic (sorted set→list, seeded RNG).
- [ ] Test: a 2-memory harness on a 20-entity toy ontology runs end-to-end and returns the result dict with both slices + AURC + novelty F1. Commit.

## Task 7: Medicine arm + driver

**Files:** Create `sma/eval/agentic/arms/medicine.py`, `scripts/agentic_suite.py`; Test `tests/agentic/test_medicine_arm.py`

- [ ] `medicine.py`: `load() -> (MountedOntology, dict[key->set])` — `mount(load_obo(hp.obo))` + parse `phenotype.hpoa` (aspect P, 7–30 phenotypes), exactly as `scripts/bench_ontology_suite.load_hpo_records`.
- [ ] `scripts/agentic_suite.py`: assemble all six memories over the Medicine arm, call `run_oneshot`, print per-memory tail (all+rare) + AURC + novelty F1 + SMA-vs-best-RAG bootstrap, write `reports/confirmatory/agentic_<arm>.csv`. Flags: `--arm medicine`, `--n-index`, `--n-query`, `--fast` (skip slow reranker).
- [ ] Test `test_medicine_arm.py`: small-N (n_index=200, n_query=20, BM25+SMA only) regression — SMA tail top-5 ≥ BM25 (sanity vs the de-risk direction). Commit.

## Task 8: Run + record (NOT a code task — an execution gate)

- [ ] Run `PYTHONHASHSEED=0 python3 scripts/agentic_suite.py --arm medicine` (full: 6 memories, 3 seeds). Append honest results to `docs/STATUS.md`: per-memory tail (all+rare), AURC, novelty F1, SMA-vs-best-RAG verdict. Commit results.

---

## Self-Review

**Spec coverage:** harness §3 → T1,T6; memories §3.1/§5 → T2-4; metrics §4 → T5; Medicine arm §7 → T7; run/record §8/§9 → T8. Interactive flagship §3 deferred to a follow-up task (noted). Generality §4 = the fact one harness runs it. ✓
**Placeholders:** none — code shown for every component; novelty threshold caveat made explicit.
**Type consistency:** `IndexItem/Query/Retrieved/Memory` used identically across T1-7; `mounted.build_case`, `mounted.config`, `mounted.canon` match `sma/ontology/mount.py`; `RetrievalResult.score/.case_id` flagged for verification in T2.
