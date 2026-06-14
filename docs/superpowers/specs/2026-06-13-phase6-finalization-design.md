# Phase 6 Finalization — Parallel Multi-Agent Orchestration (2026-06-13)

**Goal:** bring SMA-1 to Nature Machine Intelligence submission quality and complete
the blueprint's release stage, executed by **7 parallel agents** with **disjoint
file ownership** and a **fixed interface contract** so they cannot conflict.

## Shared context (every agent reads this)

- **Paper spine:** a structure-mapping memory grounded in a golden-domain ontology
  turns a generalist LLM into a verifiable specialist — it retrieves by logical
  structure (subsumption + higher-order relations) that vector RAG and KGs discard,
  beating both on rare / cross-vocabulary / high-stakes reasoning, with provenance,
  abstention, and novelty detection RAG cannot provide.
- **Results are FACTS, never invented.** Every number must trace to a committed
  `reports/confirmatory/*.csv` or `reports/calibration_grid.csv`. Headlines:
  5-domain agentic wins vs enterprise RAG/KG (medicine +0.333, genomics +0.156,
  finance +0.167, cyber +0.073, legal +0.064 tail top-5, all Holm-sig); Phase 5
  LLM-QA verifiable specialist (accuracy 0.342 vs dense 0.100; grounding-AUROC
  0.793 vs 0.547; novelty-F1 0.789 vs 0.553; selective-acc 0.625 vs 0.500 —
  4/5 axes Holm-sig, abstain-recall a tie). Honest nulls: flat-tabular
  readmission/card-fraud; invalid 4a drift (scrapped).
- **Quality bar for figures:** `References/` (MaMMAL, operator-learning, graph-memory
  papers) — dense multi-panel, **iconographic, color-coded, schematic** (entity
  icons, colored token rows, KG node diagrams, radar/heatmap insets, multi-stage
  flows). The current `fig1_overview` (word-box TikZ flowchart) is the "garbage"
  being replaced.
- **Palette (keep consistent):** SMA teal #2E8AA6, deep teal #2E6B86, first-order
  #5AA9C4, lattice violet #9B86C4, amber #D98A3D, KG gold #E7B15A, grays #C7CCD1/
  #A7AFB6/#7E8893. Domain accents: medicine #C0504D, genomics #3E8E5A, cyber
  #4A6F8A, legal #7A5DA8, finance #C68A2E.
- **FROZEN — do NOT modify:** `sma/ontology/`, `sma/eval/agentic/`,
  `sma/eval/agentic_qa/` (tags adapter-v1, prereg-v1/v2; ADR-005/006/008). New
  domains/arms may be ADDED (ADR-008) but frozen APIs and matcher dials do not move.

## Rules for ALL agents (non-negotiable)

1. **Write ONLY the files listed under your ownership.** Read anything; write
   nothing else. If you think you need another file, STOP and report it.
2. **Do NOT `git add` / `git commit` / `git checkout`.** The orchestrator integrates
   and commits once, at the end. (Avoids index races between parallel agents.)
3. **Verify your own output** — render figures, compile LaTeX, run the relevant
   tests — and report the verification result. Evidence before claims.
4. **No invented results, no placeholders, no TODOs** in delivered artifacts.
5. Report: what you produced (paths), how you verified, and any blocker.

## Interface contract (agreed filenames — wire to these even before they exist)

| Artifact | Path |
|---|---|
| Fig 1 overview (redesigned) | `paper/figures/svg/fig1_overview_v2.{svg,pdf,png}` |
| Graphical abstract | `paper/figures/svg/fig_graphical_abstract.{svg,pdf,png}` |
| Fig 3 capabilities (conceptual) | `paper/figures/svg/fig3_capabilities.{svg,pdf,png}` |
| Supplement / Extended Data | `paper/manuscript/supplement.tex` (main `\input`s it) |
| Bibliography | `paper/manuscript/references.bib` |
| Fraud-arm results | `reports/confirmatory/agentic_fraud_elliptic.csv` |

## Agents & ownership

- **A — Conceptual figures** (skill: `gimp-inkscape` + `scientific-figures`). Owns
  `paper/figures/svg/fig1_overview_v2.*`, `fig_graphical_abstract.*`,
  `fig3_capabilities.*`. Hand-built Inkscape SVG → PDF/PNG, MaMMAL-grade. No `.tex`.
- **B — Manuscript elevation** (NMI-writing brief). Owns
  `paper/manuscript/sma_nature_mi.tex` ONLY. Elevates prose/structure/format; wires
  `\includegraphics` to the v2 figure paths, `\input{supplement}`, and a
  bibliography over `references.bib`. No figures/bib/supplement files.
- **C — Extended Data + references** (skill: `scientific-figures`). Owns
  `paper/manuscript/supplement.tex`, `paper/manuscript/references.bib`,
  `scripts/figures_ed.py`. Full per-domain metrics table, paired cross-system
  transfer redo, risk-coverage curves; ~25–40 real citations. No main `.tex`.
- **D — Repo cleanup**. Owns `.gitignore`, `docs/CLEANUP.md`, and removal of build
  intermediates (`*.xdv`, `__pycache__`, stray logs). Propose deletions in
  CLEANUP.md; do NOT mass-delete code or touch other agents' files.
- **E — Zenodo prep**. Owns `.zenodo.json`, `CITATION.cff`. Metadata only (author
  Ayaz Khan / khanayaz2727@gmail.com, Apache-2.0, keywords, related identifiers).
  Stops at ready-to-publish.
- **F — HuggingFace prep**. Owns `release/hf_space/*`, `release/SSB_dataset_card.md`,
  `release/model_card.md`. Builds on existing `release/hf_space/`. Stops at
  ready-to-publish.
- **G — Structural-fraud arm (Elliptic)**. Owns `sma/eval/fraud_elliptic/` (NEW),
  `scripts/fraud_elliptic.py`, `data/raw/elliptic/`,
  `reports/confirmatory/agentic_fraud_elliptic.{csv,log}`. Download Elliptic, encode
  each transaction's **graph neighbourhood** (predecessor/successor bitcoin flows)
  as higher-order relations + a licit/illicit typology lattice — the cross-record
  structure flat-tabular lacks — and test SMA vs RAG. Reuse the frozen adapter
  read-only. Report win OR null honestly (the paper stands either way).

## Integration (orchestrator, after agents land)

Recompile manuscript (XeLaTeX) with v2 figures + supplement + bib; visually QA every
figure against `References/`; iterate figures to the bar; run the test suite; fold a
fraud-arm WIN into the manuscript boundary paragraph if it replicates; commit once
(no self co-author). Zenodo/HF stop at artifacts + exact publish commands for the user.
