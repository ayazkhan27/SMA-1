# SMA-1 — Project Architecture Blueprint

*Generated 2026-06-12 (commit `c219a95`). A definitive, first-to-last map of
what this repository contains: every layer, every module, the dataflow that
ties them together, and the design contracts that govern change. Companion to
the design contract `structure_mapping_agentic_memory_blueprint.md` (the
"what we promised to build") — this document is "what is actually built."*

---

## 0. One-paragraph orientation

SMA-1 is a **structure-mapping agentic memory**: a retrieval-and-reasoning
system that stores experience as typed predicate-calculus **structures** (not
text, not embeddings) and retrieves by **analogy of relational shape** rather
than surface similarity. Its scientific core is a from-scratch reimplementation
of Gentner's Structure-Mapping Engine (SME) with Forbus/Gentner/Law MAC/FAC
two-stage retrieval and SAGE consolidation. An LLM sits at the *orchestration*
boundary only — it chooses tools, verbalizes results, and may propose new
encoding *rules*, but it never writes facts into memory and never participates
in extraction. The repository is a Python 3.11 package (`sma/`) plus an
evaluation harness (`sma/eval/`, `scripts/`), a Gradio UI, a paper-asset
pipeline (`paper/`), and a governance trail (`docs/`, `configs/`).

It is a **monolithic, layered, dependency-inverted** architecture: a pure
domain core (IR + matcher) with no outward dependencies, wrapped by services
(store, index, SAGE), wrapped by an agent/orchestration shell, wrapped by
delivery surfaces (CLI, FastAPI, Gradio). Everything is deterministic and
content-addressed; the case store is the single source of truth.

---

## 1. Technology stack (detected)

| Concern | Choice | Where |
|---|---|---|
| Language / runtime | Python 3.11, dataclasses + `from __future__ import annotations` | everywhere |
| Domain types | frozen `@dataclass(slots=True)` (not pydantic at runtime — kept plain for inspectability) | `sma/ir/schema.py` |
| Content addressing | BLAKE3 over canonical S-expressions | `sma/ir/schema.py:content_id` |
| Combinatorial solver | OR-Tools CP-SAT (exact MWIS), greedy fallback | `sma/match/merge_cpsat.py`, `merge_greedy.py` |
| ANN | hnswlib (cosine over hashed WL-1 vectors); pure-python fallback | `sma/index/ann.py` |
| Store | LMDB when available, deterministic file+WAL fallback | `sma/store/lmdb_store.py` |
| Log parsing | Drain-style template mining + frozen ontology rules | `sma/encoders/logs_drain.py` |
| Code parsing | tree-sitter ASTs | `sma/encoders/code_treesitter.py` |
| Baselines | rank-bm25, sentence-transformers, networkx PPR, cross-encoder rerank | `sma/eval/baselines/` |
| Orchestration LLM | local llama.cpp (Qwen) or DeepSeek API, toggleable | `sma/agent/llm.py` |
| UI / API | Gradio app; FastAPI tool server | `sma/ui/app.py`, `sma/agent/api.py` |
| Stats | bespoke paired-bootstrap / Holm / Cliff's δ | `sma/eval/stats.py` |
| Paper | Matplotlib + SciencePlots + CMasher; NeurIPS LaTeX | `scripts/make_paper_assets.py`, `paper/` |
| Tests | pytest with gate markers `gate_G0..G6`, golden fixtures | `tests/` |

**Architectural pattern:** Layered + Hexagonal (ports/adapters). The IR and
matcher are the hexagon's core; encoders, store, index, baselines, and LLM
backends are adapters plugged at the edges. Dependency direction is strictly
inward (delivery → agent → services → matcher → IR); the IR imports nothing
from above it.

---

## 2. The layer cake, top to bottom

```
┌────────────────────────────────────────────────────────────────────────┐
│ DELIVERY        sma/ui/app.py (Gradio)   sma/agent/api.py (FastAPI)      │
│                 sma/cli.py / __main__.py (Typer-style CLI)               │
├────────────────────────────────────────────────────────────────────────┤
│ ORCHESTRATION   sma/agent/  comparison.py · llm.py · service.py          │
│ (the only LLM)  policies.py (hard rules) · adapter_draft.py (rule draft) │
├────────────────────────────────────────────────────────────────────────┤
│ CONSOLIDATION   sma/sage/   pools.py · assimilate.py · probabilities.py  │
├────────────────────────────────────────────────────────────────────────┤
│ RETRIEVAL       sma/index/  macfac.py · content_vectors.py ·            │
│ (MAC/FAC)                   inverted.py · ann.py                         │
├────────────────────────────────────────────────────────────────────────┤
│ MATCHER         sma/match/  mh.py · kernels.py · conflicts.py ·          │
│ (SME core)                  merge_{greedy,cpsat}.py · ses.py · mdl.py ·  │
│                             infer.py · verifier.py · explain.py · engine │
├────────────────────────────────────────────────────────────────────────┤
│ ENCODERS        sma/encoders/  logs_drain.py · code_treesitter.py ·      │
│ (deterministic)                traces.py · structured.py · agentobs.py · │
│                                prose_tier1.py · draft_adapter.py ·       │
│                                coverage.py                               │
├────────────────────────────────────────────────────────────────────────┤
│ STORE + IR      sma/store/ (lmdb_store · wal · registry)                 │
│ (foundation)    sma/ir/    (schema · sexpr · canon · signatures)         │
└────────────────────────────────────────────────────────────────────────┘
       Crosscutting: determinism · content-addressing · provenance ·
       versioned ontology (tag ontology-v1) · frozen score (tag prereg-v1)
```

### 2.1 Foundation — `sma/ir/` (the intermediate representation)

The IR is the contract every other layer speaks. It contains **no domain
terms** — only typed terms and statements — which is what makes the system
domain-agnostic.

- **`schema.py`** — the vocabulary. `Entity(name, type)` (constants/objects);
  `Statement(functor, args, ascension)` (expression trees; args are Entities
  or nested Statements — this is what makes "higher-order relations over
  statements" representable); `Case(case_id, statements, metadata)` is the
  unit of memory. `make_case()` sorts statements into canonical order and sets
  `case_id = BLAKE3(canonical_text)` — so identical structure ⇒ identical id
  ⇒ free dedup and a free audit trail. `expressions()` hash-conses
  sub-statements (shared subtrees collapse to one), turning a case into a
  rooted DAG — the graph object on which "systematicity" is defined.
- **`sexpr.py`** — the canonical serialization (`(cause (timeout svc db)
  (retry svc db))`) and its parser. Round-trip `parse∘print = id` is gate G1.
- **`canon.py`** — the canonicalizer κ and the **predicate lattice** L.
  `canonical(functor)` maps surface functors to canonical ones;
  `compatible(f_b, f_t, delta, rho)` answers "can these two functors match by
  climbing ≤δ steps to a shared ancestor, at penalty ρ^dist?" — the
  **minimal ascension** mechanism that lets disjoint vocabularies match
  through a declared ontology. (This file is where the SSB de-circularization
  lived: the old `far_`-prefix string trick was removed so the matcher cannot
  cheat.)
- **`signatures.py`** — functor signatures (arity, kind, commutativity,
  higher-order flag).

### 2.2 Foundation — `sma/store/` (the case store)

- **`lmdb_store.py`** — `CaseStore`: `put`/`get`/`exists`/`ids`/`iter_cases`
  plus an append-only `wal.jsonl`. zstd/zlib-compressed canonical S-expressions
  keyed by content id. LMDB when present, deterministic file store otherwise —
  same interface (the port/adapter seam). `replay_wal()` reconstructs the id
  set: the store is rebuildable from its log.
- **`wal.py`**, **`registry.py`** — write-ahead log and adapter/schema registry.

### 2.3 Encoders — `sma/encoders/` (deterministic extraction, no LLM ever)

The make-or-break layer. Each encoder turns a raw artifact into a `Case` with
**byte-for-byte determinism** (identical input + adapter version ⇒ identical
bytes; enforced by golden tests, gate G3).

- **`base.py`** — `Encoder` ABC + `EncodeResult{case, warnings}`; `get_encoder(id)` registry.
- **`logs_drain.py`** — the primary encoder. Template mining → event functors;
  timestamps → `before(e_i, e_j)`; aggregation → `count`/`burst`; and the
  **frozen `EVENT_CLASS_RULES` ontology** (timeout/retry/io/memory/kernel/
  network/auth/storage/lifecycle/failure events) that supplies the
  higher-order `cause`/`enables` relations carrying systematicity. Frozen at
  git tag `ontology-v1` (hash fd345c5) — the thing that must not move so that
  cross-system transfer is a real test, not tuning.
- **`code_treesitter.py`** — ASTs → `defines`/`calls`/`imports`/`throws`.
- **`traces.py`**, **`structured.py`**, **`agentobs.py`** — stack-trace
  grammars, schema-mapped JSON/CSV, and the agent's own tool outputs (this last
  is what makes the memory *agentic* rather than a document index).
- **`prose_tier1.py`** — the flagged Tier-1 prose adapter (dependency parse +
  connective lexicon); carries `tier:1` metadata forever, excluded from
  headline claims.
- **`draft_adapter.py` + `coverage.py`** — the recursive twist: an LLM may
  *propose* extra deterministic class rules (`DraftRules`), which compose with
  the frozen encoder. Rules are data, content-addressed, "LLM-proposed,
  unreviewed"-tainted; `coverage.py` is the tripwire that flags when drafted
  rules redundantly re-cover frozen ones. Encoding stays pure keyword matching.

### 2.4 Matcher — `sma/match/` (the SME core, the scientific heart)

This is the riskiest math and the paper's main algorithmic contribution. The
pipeline (`engine.py:match_cases`) is: seed match hypotheses → close support →
build kernels → resolve conflicts → merge (CP-SAT or greedy) → score → project
inferences.

- **`mh.py`** — match-hypothesis seeding and `support_closure`. Enforces SME's
  **parallel connectivity** (argument correspondences must themselves be legal
  MHs), constant semantics (`event_type` entities match by identity; integers
  released), and per-MH ascension penalties (a previous bug compounded these
  down deep chains and punished exactly the systematic matches we want — now
  fixed).
- **`kernels.py`** — root MHs and their downward support closures (the atoms of
  interpretation).
- **`conflicts.py`** — the kernel conflict graph (which kernels can't coexist
  because they'd violate one-to-one). Gmap construction reduces to
  **Maximum-Weight Independent Set** on this graph (blueprint Lemma 1).
- **`merge_cpsat.py` / `merge_greedy.py`** — exact-anytime MWIS via OR-Tools
  (with logged optimality gap), falling back to the published O(n² log n)
  greedy. `exact_or_greedy_merge` picks by kernel count vs time budget.
- **`ses.py`** — Structural Evaluation Score by trickle-down (parents pass γ-
  weighted evidence to the children they justify → deep systems dominate flat
  ones). Plus `normalize_score` (max/min/sqrt/target variants) and
  `self_score` (the denominator that makes SES scale-free across cases).
- **`mdl.py`** — Regime B: the parameter-free MDL scorer (systematicity as
  compression). `surprisal` is the default score-v2 (SES weighted by corpus
  −log₂ p), which reduces exactly to SES when costs are unit.
- **`infer.py`** — candidate inferences: statements in the base not yet mapped,
  projected into the target with fresh skolems, tagged `:hypothetical`. **This
  is the anti-hallucination mechanism** — the memory's output is a checkable
  object with provenance, not generated text.
- **`verifier.py`** — type-checks/validates projected inferences.
- **`explain.py`** — human-readable correspondence tables (what the LLM
  verbalizes).
- **`types.py`** — `MatchConfig` (the frozen dials: scorer, normalization, γ,
  ρ, δ, kernel limits) and `GMap`/`MatchHypothesis`. **The single place the
  score function is parameterized; frozen at tag `prereg-v1`.**

### 2.5 Retrieval — `sma/index/` (certified MAC/FAC)

- **`content_vectors.py`** — `functor_vector`: canonical functor counts + WL-1
  refinement features + (with δ>0) lattice ancestor-closure features. The bag
  whose weighted histogram intersection is an **admissible upper bound** on SES
  (blueprint Lemma 2).
- **`inverted.py`** — functor→postings index and the `bound()` computation (the
  certified screening bound).
- **`ann.py`** — hnswlib cosine pre-screen (used only above 20k cases).
- **`macfac.py`** — `MacFacIndex`: the two-stage retriever. Below 20k cases the
  admissible bound orders *all* candidates directly (the Liberty haystack
  forensics showed cosine pre-screening can drop true positives the bound ranks
  first); above it, ANN trims for tractability. `retrieve()` runs **best-first
  FAC**: examine candidates in descending bound, stop when no unexamined bound
  can beat the k-th best — returning a **certified-exact top-k over the
  shortlist**. `brute_force()` is the certification oracle (gate G4).

### 2.6 Consolidation — `sma/sage/` (SAGE generalization)

- **`pools.py`** — `SagePool`: assimilate a case into the best-matching
  generalization if SESₙ ≥ θ_a, else seed a new generalization from two
  mutually-assimilable outliers, else store as outlier. Generalizations are
  **schemas** that can match a novel incident matching no single exemplar.
- **`probabilities.py`** — fact probabilities = support frequencies (counting +
  Laplace smoothing; nothing learned). Low-probability facts wear away.

### 2.7 Orchestration — `sma/agent/` (the only place an LLM lives)

- **`policies.py`** — the hard rules, enforced in code not prompts:
  `reject_free_text_facts` (content enters memory ONLY via `encode`); every
  surfaced claim must carry provenance or be labeled unsupported.
- **`service.py`** — `MemoryService`: the encode/retrieve/verify facade over
  store + index + matcher.
- **`comparison.py`** — `ComparisonFramework`: the toggleable arena driving the
  UI and eval. Six modes — `sma`, `bm25`, `dense rag`, `knowledge graph`,
  `hybrid (fused)`, `context only` — over one corpus with one LLM, so what the
  UI shows is the same mathematics as the eval CSVs. Holds the draft-adapter
  state and the cite-or-abstain message builder.
- **`llm.py`** — `LocalOrchestrator` (llama.cpp/Qwen, repeat-penalty tuned
  against looping) and `DeepSeekOrchestrator` (httpx direct; key via
  `SMA_DEEPSEEK_API_KEY`/.env). System prompt enforces analogical framing and
  cite-or-abstain.
- **`api.py`** — FastAPI tool server exposing `encode/retrieve/map/project/
  verify/store/generalize/explain` (blueprint §3.1 contract).
- **`adapter_draft.py`** — builds the residual-only, dedup-guarded prompt that
  asks the LLM to draft new class rules.

### 2.8 Delivery — `sma/ui/`, `sma/cli.py`, `sma/__main__.py`

- **`ui/app.py`** — the Gradio app: continuous-chat interface, mode toggle,
  corpus loader (load-as-one-incident option), alignment-receipt display.
- **`cli.py` / `__main__.py`** — `sma encode/retrieve/map/eval …` for
  agent-drivable and human use.

---

## 3. The evaluation harness — `sma/eval/` + `scripts/`

This is half the repository by line count (~4k lines) and is what turns the
system into a *paper*. It is logically separate from the core — the core never
imports it.

### 3.1 Benchmarks and data adapters (`sma/eval/`)
- **`ssb_generator.py` / `ssb_eval.py`** — the **Synthetic Structural
  Benchmark**: (query, analog, distractor) triples with disjoint random
  vocabularies bridged only by a generated lattice, and star-rewired
  distractors provably non-isomorphic to the query. The decisive instrument
  for "structure beats surface" (H2) under perfect ground truth.
- **`loghub.py` / `loghub_eval.py`** — HDFS/BGL/Spirit/Thunderbird/OpenStack/
  Liberty samplers with leakage discipline (label/alert columns stripped
  before encoding).
- **`family_labels.py`** — deterministic HDFS failure-family labels from
  session text.
- **`transfer_eval.py`** — the cross-system transfer harness; builds all seven
  retrieval methods (SMA + BM25 + dense + KG-PPR + HippoRAG + Hybrid-RRF +
  Hybrid+Rerank) once per index and runs the weighted-vote triage.
- **`bugsinpy.py` / `bugsinpy_families.py`** — code-domain LOPO bug retrieval.
- **`arn.py`** — the ARN narrative-analogy adapter (Tier-1).
- **`drift_env.py`** — the seeded ops-world generator for the T5 drift protocol.
- **`metrics.py`, `report.py`, `stats.py`** — macro-F1/hit@k/MRR; the
  `report.html` fixture generator; and the pre-registered statistics
  (paired bootstrap, Holm-Bonferroni, Cliff's δ).

### 3.2 Baselines (`sma/eval/baselines/`)
`bm25.py`, `dense.py`, `bge_dense.py`, `splade.py`, `wl_kernel.py`,
`hybrid_rrf.py`, `rerank.py` (cross-encoder), `hipporag.py` (HippoRAG-2-style
phrase-graph + personalized PageRank, deterministic extraction substituted for
LLM OpenIE), `longcontext_llm.py`. Each shares SMA's Tier-0 *facts* where
applicable so the comparison isolates the *memory mathematics*.

### 3.3 Driver scripts (`scripts/`)
`calibrate.py` (the validation grid), `confirmatory_battery.py` (the
single-shot, seed-locked test runner), `scorer_gauntlet.py`,
`transfer_controls.py`, `family_eval.py`, `baseline_ladder.py`,
`bugsinpy_eval.py`, `h3_mini_study.py`, `prepare_ui_corpus.py`,
`make_paper_assets.py`, `fetch_datasets.py`, `fetch_model.py`.

---

## 4. Dataflow — the two end-to-end paths

**WRITE path (how experience enters memory):**
```
raw artifact ─▶ Encoder (deterministic) ─▶ Case (s-expr, BLAKE3 id)
            ─▶ CaseStore.put + WAL ─▶ MacFacIndex.add (content vector → ANN + inverted)
```

**READ path (how a question is answered):**
```
query artifact ─▶ Encoder ─▶ query Case
   ─▶ MAC: content vector → bound-ordered shortlist (certified)
   ─▶ FAC: SME match_cases on each candidate → SESₙ (best-first, early-stop)
   ─▶ top-k analogs + alignment receipts
   ─▶ candidate_inferences (projected, :hypothetical, with provenance)
   ─▶ verifier
   ─▶ LLM orchestrator: verbalize from receipts — CITE OR ABSTAIN
```

The LLM touches only the last step. Every claim it surfaces traces to a stored
case and a specific set of correspondences.

---

## 5. Crosscutting concerns

- **Determinism** — every encoder is byte-reproducible; samplers use `sorted()`
  and fixed seeds; `Date.now`/`random` avoided in reproducible paths. Enforced
  by golden fixtures (`tests/golden/`).
- **Provenance & anti-hallucination** — content-addressed cases; candidate
  inferences carry `{base_id, gmap_id, SESₙ, support, skolems, ascensions}`;
  policy layer rejects free-text facts.
- **Versioning as governance** — two frozen artifacts gate scientific validity:
  `ontology-v1` (the encoder rules, frozen before Spirit was first read) and
  `prereg-v1` (the score function + test protocol, frozen before any test-seed
  run). ADRs (`docs/ADR/001–006`) record every change to §2 math or §8 design;
  STATUS.md is the append-only ledger.
- **Error handling** — optional heavy deps (LMDB, hnswlib, sentence-
  transformers, tree-sitter) degrade to deterministic fallbacks behind the
  same interface, never crash the core.
- **Configuration** — `configs/default.toml`, `model.toml`,
  `preregistration.md` (the frozen dials + protocol).

---

## 6. Testing architecture

`tests/` mirrors the blueprint's gate structure: `test_gates.py` carries
pytest markers `gate_G0..G6` (G0 bootstrap/datasets, G1 IR round-trip + store,
G2 matcher canonical battery [water-flow↔heat-flow, solar-system↔atom], G3
encoder determinism, G4 certified-MAC/FAC == brute force + lattice-bridging, G5
agent policies + provenance, G6 SAGE probabilities). Plus `test_stats.py`
(statistics), `test_hipporag.py` (KG baseline), `test_bugsinpy_t3.py`,
`test_production_loop.py`. Golden fixtures in `tests/golden/` pin byte-exact
encoder output and canonical SME mappings. Full suite is currently 50 tests,
green.

---

## 7. How to extend (the blueprint for new work)

- **New domain** → write a new deterministic encoder in `sma/encoders/`
  implementing `Encoder.encode → EncodeResult`, register it, add golden
  fixtures. The IR and matcher do not change. (Optionally declare a per-domain
  mini-ontology in `canon.py`'s lattice.)
- **New baseline** → add to `sma/eval/baselines/` with the
  `build(documents)`/`retrieve(query, k)` shape; wire into the
  `transfer_eval` retrievers dict (picked up dynamically by the battery).
- **New benchmark/task** → add a `sma/eval/<task>.py` + a `scripts/` driver +
  a `confirmatory_battery.py` task function; emit per-query rows so the stats
  module can run on it.
- **Score change** → forbidden without a new `score-vN`: edit `MatchConfig`
  defaults, re-run G2 + the calibration grid + the sensitivity sweep, write an
  ADR, and cut a new prereg tag. The freeze is load-bearing.
- **New figure/table** → add to `make_paper_assets.py` (single source of
  truth; uniqueness guards forbid duplicate titles/filenames); never hand-type
  numbers — they come from `reports/` CSVs.

---

## 8. Architectural decision records (where the "why" lives)

- **ADR-001** stack & scope · **ADR-002** goal acceptance · **ADR-003**
  retrieval & orchestration policy · **ADR-004** scorer ablation SES vs MDL ·
  **ADR-005** score-v2 surprisal-SES · **ADR-006** matcher semantics v4
  (constants, parallel connectivity, bound-ordered MAC, max-normalization).
  Read these before touching scoring or matcher semantics.

---

*This blueprint reflects the repository at commit `c219a95`. Regenerate the
architectural map after any phase that adds a layer, encoder, or baseline.*
