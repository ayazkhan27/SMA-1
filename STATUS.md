# SMA-1: Structure-Mapping Agentic Memory - Implementation Status

Snapshot of the current state. The append-only session ledger lives in
[docs/STATUS.md](docs/STATUS.md); the design contract is
[structure_mapping_agentic_memory_blueprint.md](structure_mapping_agentic_memory_blueprint.md).

## Current State (2026-06-11)

- All 11 test gates (G0-G6) pass: `make test`.
- LogHub MVP diagnostic evaluation **executed end-to-end**: `make report`
  (runs SSB fixtures + full HDFS/BGL; `--skip-loghub` for fixtures only).
- Gradio comparison workbench live: `make ui` → http://127.0.0.1:7860
  (toggleable side-by-side modes: sma / bm25 / dense rag / knowledge graph / context only).

## LogHub MVP Diagnostic Results (1,000 sessions/dataset, seed 42)

| Dataset | Method | Macro-F1 | hit@1 | hit@5 | hit@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|
| HDFS | **SMA** | **0.9549** | 0.9550 | 0.9090 | 0.8770 | 346 | 638 |
| HDFS | BM25 | 0.8191 | 0.7550 | 0.7450 | 0.7105 | 25 | 38 |
| HDFS | Dense RAG | 0.6449 | 0.6400 | 0.6160 | 0.6170 | 23 | 26 |
| HDFS | KG-PPR proxy | 0.6625 | 0.6800 | 0.6850 | 0.6765 | 6 | 8 |
| BGL | SMA | 0.8687 | 0.8700 | 0.8690 | 0.8645 | 293 | 2464 |
| BGL | BM25 | 0.9750 | 0.9700 | 0.9600 | 0.9620 | 12 | 437 |
| BGL | **Dense RAG** | **0.9950** | 0.9950 | 0.9920 | 0.9815 | 26 | 369 |
| BGL | KG-PPR proxy | 0.8895 | 0.9700 | 0.9040 | 0.8945 | 4 | 46 |

No diagnostic alert rows (no baseline collapse, no suspicious perfection flags).

**Honest reading.** SMA wins decisively within-system on HDFS (+13.6 F1 points over
the best baseline). It trails on BGL, where anomalous messages carry overt lexical
markers, making surface retrieval near-perfect — H2 (within-domain non-inferiority)
currently fails on BGL and is reported as a finding, not hidden. The blueprint's
primary claim (H1) concerns *cross-system / off-surface* transfer, which has not
been run yet.

## Matcher Performance (this session's headline change)

Found via a live py-spy stack dump: `Kernel.bindings` was rebuilt (with full
statement re-serialization) on every access inside the merge inner loop, and
repeated event types exploded the kernel count quadratically. Fixes:

- Cached binding tables and MH keys (`cached_property`).
- MH seeding capped per canonical functor group (`MatchConfig.mh_group_cap=128`,
  U-ordered, identical statements first) — the blueprint §10.2 tripwire response.
- Root-MH-only kernels per §2.2.

Pathological 120-line session match: **>5 min → ~181 ms (~2000×)**. The §2.8
target (≤50 ms median at ~300 statements) is met. Canonical battery (G2) re-verified.

## Key Commands

```
make test     # all gates
make report   # full eval (unbuffered, streams progress)
make ui       # Gradio workbench
make api      # FastAPI tool endpoints
```

## Next Actions

1. Cross-system transfer eval (BGL→Thunderbird, HDFS→OpenStack) — the real H1 test.
2. Investigate the BGL gap (per-message anomaly markers vs structural signal in the
   60 s window encoder; BGL p95 latency 2 s on large windows).
3. Consolidated pre-freeze eval batch: hybrid RRF + reranker baselines, multi-seed transfer confirmation (incl. MDL leg), long-context B6 designed against H3 findings; then calibration freeze + prereg tag.
