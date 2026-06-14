---
title: SMA-1 Structure-Mapping Agentic Memory
emoji: 🧩
colorFrom: teal
colorTo: indigo
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: Retrieval by relational structure — cite, abstain, detect novelty.
---

# SMA-1 — Structure-Mapping Agentic Memory (demo Space)

**One universal structure-mapping memory beats enterprise RAG/KG across five
golden-ontology domains — with calibrated abstention and novelty detection
that vector RAG structurally cannot provide.**

This Space demonstrates the SMA-1 system from the paper
*"Structure-Mapping Agentic Memory"* by Ayaz Khan (2026, under review at
*Nature Machine Intelligence*).

## What the demo shows

Toggle between three clinical phenotype queries to see **SMA structure-mapping
retrieval vs dense vector-RAG** side by side:

| Feature | SMA | Dense vector-RAG |
|---|---|---|
| Retrieval basis | Relational structure (subsumption lattice, higher-order relations) | Term co-occurrence vector (cosine similarity) |
| Cite-or-abstain calibration | AUROC 0.793 | AUROC 0.547 (near chance) |
| Novelty detection | SAGE expectation-violation, F1 0.789 | None (F1 0.553 — threshold artefact) |
| Cross-vocabulary transfer | Lattice ascension bridges disjoint vocabularies | Fails without shared surface vocabulary |

*Metrics from `reports/confirmatory/qa_stats.csv`, Phase 5 prereg-v2 run.*

## Verified headline numbers

Five unrelated domains, all Holm-significant wins vs best enterprise RAG (tail top-5):

| Domain | Ontology | SMA Δ vs best enterprise RAG |
|---|---|---|
| Medicine | HPO | **+0.333** (p = 0.0002) |
| Genomics | GO | **+0.156** (p = 0.0002) |
| Finance | US-GAAP | **+0.167** (p = 0.0002) |
| Cybersecurity | ATT&CK | **+0.073** (p = 0.035) |
| Legal | CPC | **+0.064** (p = 0.0022) |

Structure Synthesis Benchmark (zero-lexical-overlap forced-choice):
**SMA r@1 = 1.000 vs BM25 = 0.000, Dense = 0.000.**

## About the demo scores

The retrieval scores in the comparison tab are **pre-computed illustrative examples**
from the frozen `adapter-v1` evaluation.  This Space does not re-run the full SMA
engine at inference time (which requires the `sma` package + ontology files); instead
it shows representative outputs clearly labelled as pre-computed.

The verified numbers shown in the *Metrics* tab trace directly to committed
`reports/confirmatory/*.csv` files in the source repository.

## Running locally

```bash
git clone https://github.com/ayazkhan27/sma-1
cd sma-1
pip install -e ".[eval]"
python release/hf_space/app.py          # illustrative mode (no ontology files needed)
# OR
make ui                                 # full live engine (requires datasets)
```

## Resources

- **Source code + paper:** https://github.com/ayazkhan27/sma-1
- **arXiv preprint:** (link pending)
- **Zenodo artifact DOI:** (link pending)
- **Dataset card (SSB):** https://huggingface.co/datasets/ayazkhan27/sma-ssb
- **Model card (adapter-v1):** https://huggingface.co/ayazkhan27/sma-adapter-v1
- **License:** Apache-2.0

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

> Built by the `scripts/build_release.py` pipeline; do not edit in-Space —
> edit the source repo and re-sync.  No GPU required.  Any LLM backend key
> (for the full live engine) is supplied via **Space secrets**, never committed.
