---
title: SMA-1 Structure-Mapping Agentic Memory
emoji: 🧩
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.0.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Retrieval by relational structure, not surface similarity.
---

# SMA-1 — Structure-Mapping Agentic Memory (demo Space)

Toggle between **SMA** (certified MAC/FAC retrieval + SME structure mapping with
provenance-tagged candidate inferences) and **BM25 / dense RAG / knowledge-graph
/ hybrid / context-only** over one corpus, with one LLM, so you see the same
retrieval mathematics that produced the paper's results.

This Space wraps `sma.ui.app`. The frozen configuration (score-v2: surprisal ×
max, γ=0.25, ρ=0.95; tag `prereg-v1`) is the default; other scorers are marked
*exploratory*.

- Code / paper: <GitHub URL>  ·  arXiv: <id>  ·  License: Apache-2.0
- No GPU required. Any LLM backend key is supplied via **Space secrets**, never
  committed.

> Built by the `scripts/build_release.py` pipeline; do not edit in-Space — edit
> the source repo and re-sync.
