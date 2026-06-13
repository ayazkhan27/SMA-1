# Phase 4a — Drift design spec

Date: 2026-06-12 · Status: **awaiting maintainer approval** · Supersedes blueprint §8.3 T5 (synthetic ops world)

## 1. Goal & hypothesis

Demonstrate the agentic-memory differentiator: **structural memory that is
re-derived from the environment resists drift, while memories built from the
agent's own generations compound error.** Plus a unique detector: **SAGE
expectation-violation flags concept drift structurally** — something no
embedding/graph store does.

- **H4 (state/data drift):** as the interaction horizon grows and facts evolve,
  SMA's answer accuracy on drift-sensitive questions decays more slowly than
  generative memories.
- **H6 (concept drift detection):** when a fact's value changes (a knowledge
  update), SMA's SAGE pool fires an expectation-violation; we score it as a
  drift detector (detection delay, recovery) against the standard protocol.

## 2. Arena & framing (decision: agent-memory benchmark, real data)

**Benchmark:** **LongMemEval** (primary) — its *knowledge-updates* category is
concept drift by construction (a user fact changes across sessions; the memory
must return the *current* value), and *temporal-reasoning* tests event
timelines. **LoCoMo** secondary (long multi-session temporal recall). Both are
real (human-machine constructed dialogues, human-edited), not synthetic.

**Extraction = LLM-based, held constant (honest framing).** LongMemEval is
free-text dialogue; deterministic Tier-0 encoding does not apply. *Every*
agent-memory system on this leaderboard (Zep, Mem0, A-MEM) uses LLM extraction,
so SMA using LLM extraction here is **matching the field's setup, not abandoning
our identity** — it makes the comparison the fairest possible: extraction is
identical across variants, so the experiment isolates purely the **memory
mechanism**. The deterministic-extraction, $0-token, auditable story remains the
headline in the log / code / (future) process-mining domains; this arena is
explicitly captioned "isolating the memory mechanism."

## 3. The four memory variants (the only thing that varies)

Shared: DeepSeek as both the **extraction** model and the **answer** model; the
same retrieval-then-answer loop. Variants differ only in the memory substrate.

| Variant | Substrate | Drift behavior |
|---|---|---|
| **context-only** (control) | running text summary in-context | compounds: prose feeds forward |
| **RAG-notes** | LLM-written notes, dense-retrieved | compounds: notes are generated text, go stale |
| **Zep / Graphiti** (SOTA) | temporal knowledge graph, validity-windowed facts; **run by us on DeepSeek** for an equal-footing head-to-head | bi-temporal model designed for evolving facts |
| **SMA** (ours) | cases re-encoded from each turn's extracted structure; SME structural retrieval; SAGE pools | re-derived per turn; SAGE flags schema violations |

A common `MemoryBackend` interface — `ingest(session_events)`,
`query(question) -> (answer, retrieved, drift_flag)` — with four
implementations, so each is testable in isolation and swappable.

## 4. Data flow

```
LongMemEval sample = multi-session history H + questions Q
  ingest:  for each session turn -> DeepSeek extracts facts/triples
           -> each backend writes them to its own substrate
  query:   for each q in Q -> backend.retrieve -> DeepSeek answers from retrieval
           -> score answer vs gold (LongMemEval grader)
  drift:   knowledge-update q's are tagged; we additionally measure update-recovery
           and (SMA) SAGE expectation-violation vs the change point
```

## 5. Metrics (reconciles agent-memory eval + standard drift protocol)

1. **LongMemEval accuracy**, overall and **per category** — headline on
   *knowledge-updates* and *temporal-reasoning* (the drift-sensitive ones).
2. **Update-recovery / staleness:** for each knowledge-update (fact f: v1→v2 at
   session s), did the memory return v2 (current) not v1 (stale) after s, and
   how many sessions later did it switch? = the **detection-delay / recovery**
   analog from the standard concept-drift protocol (prequential test-then-learn,
   ADWIN-style recovery).
3. **SAGE expectation-violation as a drift detector (SMA-unique):** scored with
   the standard detector metrics — detection delay, precision/recall of
   flagged-change vs true change points — against the frequency baselines.
4. Stats: same single-shot discipline — multiple LongMemEval slices/seeds,
   paired bootstrap + Holm + Cliff's δ. Outputs `reports/confirmatory/t5_*.csv`.

## 6. Components (isolation boundaries)

- `sma/eval/longmemeval.py` — loader + grader for the benchmark (fetched by
  checksum manifest; data NOT vendored).
- `sma/eval/memory_backends/` — `base.py` (the `MemoryBackend` interface) +
  `context_only.py`, `rag_notes.py`, `zep_graphiti.py`, `sma_memory.py`.
- `sma/sage/` — extend with `expectation_violation()` on the existing pools
  (`sma/sage/pools.py`) — the drift detector.
- `scripts/drift_battery.py` — the harness (single-shot, seeded, the four
  variants), emitting `t5_*.csv`; mirrors `confirmatory_battery.py` discipline.
- `sma/eval/stats.py` — reused for the drift stats.

## 7. Cost, scale, dependencies

- DeepSeek budget: ~$14 available (operator will top up). LongMemEval ≈ 500
  questions over long histories; extraction is the token driver. We pilot on a
  small slice first, log projected cost, then run the full single-shot.
- Zep/Graphiti needs a graph DB (Neo4j or FalkorDB) + its extraction pipeline,
  pointed at DeepSeek. This is the heaviest dependency; stand it up in a
  container, isolated from the core (the core never imports it).
- All four variants share the DeepSeek backbone for a clean head-to-head.

## 8. Honest caveats (baked into the paper framing)

- Extraction here is LLM-based, *by design and by field norm* — not our
  deterministic headline; clearly captioned.
- Real run-to-run LLM variance; absorbed by seeds + bootstrap.
- SMA may be **parity** with Zep rather than a win — Zep is purpose-built for
  temporal facts. We report honestly; even parity-with-SOTA *plus* the unique
  SAGE structural drift-detector is a strong result.

## 9. Success criteria

- The harness runs single-shot over LongMemEval for all four variants with
  per-category accuracy + drift metrics, reproducible from a seed + the CSVs.
- H4: SMA's knowledge-update/temporal accuracy decays slower than context-only
  and RAG-notes (corrected CI excludes 0 in the predicted direction).
- H6: SAGE expectation-violation detects knowledge-update change points with
  measurable delay/recall vs the frequency baseline.
- Outcome reported whether positive, parity, or negative (honest-outcome rule).

## 10. Out of scope (YAGNI)

- No new deterministic encoders here (that's 4b / process-mining).
- No latency optimization (Phase 5).
- LoCoMo is secondary; ship LongMemEval first, add LoCoMo only if budget allows.
