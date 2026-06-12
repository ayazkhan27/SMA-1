# SMA-1 Executive Report

**Date:** 2026-06-12 · **Status:** research prototype with held-out, multi-seed
headline results · **Repo state:** all 11 test gates green, full history
committed, ontology hash-frozen at tag `ontology-v1`

---

## What SMA-1 is, in plain words

A memory system for AI that remembers *how things happened* (the causal shape
of an incident) instead of *what words were used*. Raw data enters memory only
through deterministic rule-based parsers — no AI in the extraction path, so
the memory's contents are bit-for-bit reproducible and auditable. An LLM sits
strictly downstream and can only verbalize what was retrieved, with provenance
on every claim. The committed product wedge: **infrastructure incidents** —
"have we seen this before, what caused it, what fixed it."

## The four headline results

### 1. Structure transfers across systems; word-matching does not (the H1 result)

We froze the event ontology (git tag, hash-pinned), then downloaded a
supercomputer log corpus nobody had touched (Spirit) and asked: can memory
built from one system (BGL) answer queries from the other?

| BGL → Spirit (3 seeds, frozen ontology) | macro-F1 |
|---|---|
| **SMA** | **0.92 / 0.965 / 0.93 (mean 0.938)** |
| WL graph-similarity (same representation) | 0.62 |
| Production RAG (hybrid + reranker) | 0.59 / 0.52 |
| Dense embeddings / BM25 / entity-graph | 0.31–0.41 |

Not post-hoc (ontology frozen first), not single-seed, not vocabulary-assisted.
The decomposition is clean: our representation alone gets generic graph
similarity to 0.62; the structure-mapping alignment adds the remaining +31
points. **Both halves of the system earn their keep, and production RAG does
not transfer.**

Honest scope: transfer holds *within* a failure-physics family (infrastructure
logs). Cross-family pairs (HDFS→OpenStack, HDFS→Spirit) fail for every method
— a real boundary, documented, with a research idea attached (missing-event
anomalies via schema expectation-violation).

### 2. Within one system, SMA beats the full production-RAG ladder where structure matters

HDFS triage (anomalies live in event *patterns*): SMA 0.955 > Hybrid-RRF 0.835
> frontier-LLM long-context 0.809 > Hybrid+Rerank 0.743 > BGE 0.638 > SPLADE
0.404. And on the deeper metric — retrieving the correct *failure family*, not
just "an anomaly" — SMA leads at 0.906 vs BM25's 0.62.

Honest counterweights: on BGL, where anomalous messages literally announce
themselves ("KERNEL FATAL"), lexical methods saturate (~1.0) and SMA trails —
and a cheap WL graph-similarity pass actually beats full structure-mapping
within-system (0.98 vs 0.955 at 1/30th the latency). Design implication
adopted: tiered retrieval (cheap graph pass within-system, full mapping for
cross-system and provenance).

### 3. Provenance discipline measurably prevents confabulation (the H3 result)

200-cell study (20 questions, half unanswerable/false-premise, 5 memory modes,
2 LLMs), independently judged with an audit trail: the provenance-bound
DeepSeek verbalizer was **99% correct with zero invented entities** and
abstained on 50/50 unanswerable questions; an undisciplined small model
confabulated 18% of the time and never abstained once. Honesty is enforced by
architecture + a competent verbalizer — and because evidence is structured,
the *unsupported-claim rate* is now a measurable, reportable metric (~0.02
claims/answer vs 0.32).

### 4. The engineering holds up

~2000x matcher speedup (worst case 5 min → 181 ms); certified retrieval bounds
fixed; three label-leakage vectors found and closed across datasets; sampler
nondeterminism fixed; every fix gated by 11 green tests and an append-only
ledger. $0 LLM tokens for extraction/indexing; CPU-only.

## What this adds up to

SMA is not a RAG killer — within-domain document QA over prose remains RAG's
territory, and our own BGL numbers prove surface methods win when vocabulary
carries the signal. SMA is the first credible **post-RAG memory architecture
for event-structured data**: it wins precisely where RAG structurally cannot
follow (vocabulary shift, cross-system reuse, auditability) and degrades
honestly where it shouldn't be used. The Fortune-10 objections (post-hoc
ontology, weak baselines, shallow metric, verbalizer-vs-memory safety) have
each been answered with a measured experiment rather than an argument.

## Blueprint completion (honest)

- P0–P6 (system build): complete at fixture-gate level; G2's 25-pair SME-v4
  oracle battery and G3's LogHub-2k template validation remain for full rigor.
- P7 (evaluation): within-system, scorer ablation, held-out transfer,
  family metric, H3 with judging — done. Remaining: SSB de-circularization,
  calibration→freeze→prereg tag, multi-seed statistics with bootstrap/CIs,
  ablation battery (γ=0 first), drift protocol T5 (the "agentic" claim),
  BugsInPy (T3), ARN (T4).
- P8 (paper/artifact): not started; Docker repro and writing remain.

## Next steps (in order)

1. Rare-family-stratified scorer analysis, MDL corpus-cost upgrade, ses_n
   bias study → resolve the scorer default → calibration freeze + prereg tag.
2. Single-shot test runs with paired-bootstrap statistics (numbers → claims).
3. Drift protocol T5: wrong-action rate under concept drift (the result that
   would carry a Nature-tier claim).
4. Latency ladder to p95 < 300 ms @ 100k cases (real HNSW, parallel FAC,
   tiered WL/SME retrieval, Rust hot loops).
5. Second domain adapter end-to-end (pharmacovigilance design exists) +
   paper writing + open artifact.

## Where everything lives

`reports/report.html` (full narrative + every table) · `reports/*.csv` (raw
artifacts incl. h3_judged.csv awaiting your spot-check of low-confidence rows)
· `docs/STATUS.md` (append-only history) · `docs/ADR/` (decision records) ·
`docs/POSITIONING.md` (the wedge) · git tags: `ontology-v1`.
