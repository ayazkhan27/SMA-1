# SMA-1: Structure-Mapping Agentic Memory

This repository is a minimum-viable research artifact for the blueprint in
`structure_mapping_agentic_memory_blueprint.md`.

The MVP provides:

- deterministic typed S-expression cases,
- a lightweight append-only case store,
- an SME-style matcher with support closure, kernel merge, SES/MDL scoring, and candidate inference,
- MAC/FAC-style retrieval with an admissible histogram bound,
- deterministic Tier-0 encoders for logs, traces, code, structured data, and agent observations,
- SAGE-style consolidation,
- FastAPI and Gradio entrypoints,
- reproducible dataset acquisition manifests and report generation.

## Quick Start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[encoders,eval]"
pytest
sma ui
```

For a minimal dependency smoke run:

```bash
python -m pytest -m "gate_G0 or gate_G1 or gate_G2 or gate_G3 or gate_G4 or gate_G5 or gate_G6"
python -m sma.eval.report --out reports/report.html
```

Raw datasets are not tracked. Use `scripts/fetch_datasets.py --manifest data/manifests/datasets.json`
to download checksum-verified external data into `data/raw/`.

