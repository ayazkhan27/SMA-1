# Repository Cleanup Inventory (Phase 6, 2026-06-13)

Agent D has identified and catalogued stale, superseded, or intermediate artifacts. This document lists candidates for archival or deletion with per-item recommendations. **The orchestrator makes the final deletion decision; this agent does not delete.**

## Summary of Actions Taken

- Updated `.gitignore` to ignore XeLaTeX `.xdv` output, intermediate `.bcf`, `.run.xml`, `.synctex.gz` files, and Python bytecode `.pyo`, plus backup `.bak` and temporary `.tmp` files.
- Removed **all untracked build intermediates**: 17 LaTeX `.aux` and `.log` files from `paper/manuscript/`, `paper/figures/tikz/`, and `paper/figures/individual/`; 21 `__pycache__` directories across the entire project tree.
- Verified 5 untracked CSV/log files remain (see below).

---

## Untracked Intermediate Outputs

These files exist in the working tree but are untracked by git. **Status:** not removed; awaiting orchestrator decision.

### `reports/confirmatory/t5_rows.csv` (69 KB)
- **Content:** 1500 rows of temporal-reasoning task results (gpt4 question IDs, category, method, correctness, drift flag, novelty flag).
- **Context:** Output from Phase 4a drift-run experiment (commit `01a753c`), marked INVALID for SMA (strawman encoder).
- **Current use:** None; superseded by Phase 5 agentic results (medicine/cyber/finance/legal/discovery arms).
- **Recommendation:** **DELETE** — Phase 4a is superseded; no reference in current manuscript or confirmed reports.

### `reports/confirmatory/t5_stats.csv` (281 bytes)
- **Content:** Summary statistics (delta, CI, Cliff's d, Holm-corrected p-values) for t5 results vs context-only and rag-notes baselines.
- **Context:** Derived from `t5_rows.csv` during invalid Phase 4a drift run.
- **Recommendation:** **DELETE** — Paired with invalid `t5_rows.csv`; no current use.

### `reports/confirmatory/cd_diabetes_before.csv` (251 bytes)
- **Content:** Single row: phase=3, domain="generic", baseline="cd_before", HO_density=0, metrics (SMA F1 0.775 vs baseline 0.620, delta=+0.155).
- **Context:** Phase 4b diabetes-130 cross-domain experiment (commit `0eef80c`), generic (non-domain-specific) adapter evaluation before full cross-domain design.
- **Current use:** None; Phase 4b was a development experiment; Phase 5 replaced with curated 5-domain agentic arms (medicine, cyber, finance, legal, discovery).
- **Recommendation:** **DELETE** — Phase 4b is exploratory; no reference in final paper or confirmed reports.

### `reports/confirmatory/ontology_hpo_regression.log` (371 bytes)
- **Content:** Regression-test output for rare-disease diagnosis (200 simulated patients, HPO generic ontology API): SMA top-5 0.880, Phenomizer 0.810, Jaccard 0.605; marked "REGRESSION PASS".
- **Context:** Validation step during Phase 3/4 ontology-adapter development (no commit found; likely from an earlier run or intermediate check).
- **Current use:** None; Phase 5 replaces with registered agentic harness results on real OMIM dataset.
- **Recommendation:** **DELETE** — Superseded by registered Phase 5 results; regression test was developmental, not part of final evidence.

---

## Historical Figure Artifacts

All tracked figures below are **intentionally preserved** in the repository as historical record and reference for future iterations. They are not referenced by the current manuscript (`sma_nature_mi.tex`).

### `paper/figures/*.pdf` and `paper/figures/*.png` (root level)
- **Files:** `fig1_sma_overview.{pdf,png}`, `figS1_t1_statistics.{pdf,png}`, `fig_decomposition.{pdf,png}`, `fig_family_scorers.{pdf,png}`, `fig_h3_honesty.{pdf,png}`, `fig_ladder_hdfs.{pdf,png}`, `fig_pipeline_overview.{pdf,png}`, `fig_structure_mapping.{pdf,png}`, `fig_transfer_headline.{pdf,png}`.
- **Content:** High-level conceptual and results figures from earlier paper drafts and exploratory analyses.
- **Status:** All are **tracked** (committed); not part of Phase 6 cleanup scope.
- **Recommendation:** **KEEP** — Tracked historical assets; useful reference for future revisions and supplementary material development.

### `paper/figures/individual/fig0[1-4]_*.{pdf,png,tex}` and `fig05-13_*.{pdf,png}`
- **Files:** TikZ-compiled individual figures (architecture, representation, matcher, mechanism, transfer, SSB, family, triage, code, certified, calibration, haystack, radar).
- **Status:** All **tracked**; represent the "old figure design" library before Phase 6 redesign to SVG/Inkscape.
- **Context:** Phase 5 and earlier figure iterations; superseded by Agent A's Phase 6 Inkscape SVG redesigns (fig1_overview_v2, graphical abstract, fig3_capabilities).
- **Recommendation:** **KEEP FOR NOW, TAG FOR ARCHIVAL** — These TikZ figures are not used by the current manuscript but are valuable as reference and diff history for the design overhaul. If manuscript moves fully to Inkscape-only workflow, consider archiving to a `paper/figures_archive/` subdirectory in a future cleanup.

### `paper/figures/individual/fig1_overview-1.png` (94 KB)
- **Content:** Intermediate render of `fig1_overview.tex` (older TikZ iteration of the universal-adapter diagram).
- **Status:** Tracked; generated artifact from `fig1_overview.tex`.
- **Recommendation:** **KEEP** — Part of tracked figure set; useful for diff history and design evolution narrative.

### `paper/figures/individual/fig_agentic_results.{pdf,png}` (46 KB, 480 KB)
- **Content:** Results visualization from Phase 5 agentic arm runs.
- **Status:** Tracked; historically important but not used in current manuscript (replaced by `svg/figure2_results.{pdf,png}` and `svg/figure5_trustworthy.{pdf,png}`).
- **Recommendation:** **KEEP** — Tracked; historical reference for agentic benchmarking visualization evolution.

---

## Current Manuscript Figure References

These figures **are used** by `paper/manuscript/sma_nature_mi.tex` and must not be deleted:
- `paper/figures/individual/fig1_overview.pdf` (referenced at line 88)
- `paper/figures/svg/figure2_results.pdf` (referenced at line 152)
- `paper/figures/svg/figure5_trustworthy.pdf` (referenced at line 187)

Corresponding PNG and SVG files should be retained for reproducibility and re-export.

---

## Untracked Directories

### `ClaudeDesignBadFigures/` (directory)
- **Content:** Early Inkscape SVG drafts for figures 1, 3, and 4 (5 files: "Figure 1a", "1b", "1c", "Figure 3", "Figure 4").
- **Status:** Untracked; created during Agent A's design iteration.
- **Purpose:** Exploratory designs before final versions.
- **Recommendation:** **ARCHIVE OR DELETE** — These are tagged as "bad" and represent rejected design iterations. Agent A should confirm they are no longer needed before deletion. For now, untracked so safe to remove without git history loss. Suggest archival to `paper/design_archive/` if future design reference is valued.

### `References/` (directory)
- **Content:** Untracked directory with 4 PNG files (MaMMAL figure collection, pasted notes from reference papers).
- **Status:** Untracked; appears to be a working notes/reference collection for Agent A's figure design work.
- **Purpose:** Visual reference and inspiration for figure redesign.
- **Recommendation:** **KEEP IF ACTIVE, OTHERWISE ARCHIVE** — If Agent A is still iterating on figures, keep. Otherwise, consider archival to `paper/design_archive/references/` to reduce clutter.

### `paper/figures/prompts/` (directory)
- **Content:** Claude Design prompts used for figure generation (per-figure instructions).
- **Status:** Tracked; part of the design workflow documentation.
- **Recommendation:** **KEEP** — Useful for understanding how figures were generated and for future iterations.

---

## XeLaTeX Build Artifacts (Removed)

The following **untracked** XeLaTeX compilation outputs have been **removed** during Phase 6 cleanup:

| File | Type | Reason |
|------|------|--------|
| `paper/manuscript/sma_nature_mi.xdv` | XeLaTeX DVI (XeTeX) | Untracked intermediate; regenerated on each compile |
| `paper/manuscript/main.aux`, `sma_nature_mi.aux` | LaTeX auxiliary | Untracked intermediates; regenerated on each compile |
| `paper/manuscript/main.log`, `sma_nature_mi.log` | LaTeX log | Untracked intermediates; regenerated on each compile |
| `paper/figures/tikz/fig1.aux`, `fig1.log` | LaTeX auxiliary + log | Untracked intermediates from TikZ figure builds |
| `paper/figures/individual/fig0[1-4]_*.aux`, `*.log` | LaTeX auxiliary + log | Untracked intermediates from individual figure builds |

**Proof of removal:** All confirmed untracked via `git ls-files --error-unmatch <path>` (exit code 1) before deletion.

---

## Python Cache (Removed)

**21 `__pycache__` directories** have been **removed** from:
- `tests/`, `sma/`, `scripts/`, and all submodules
- `sma/ontology/`, `sma/eval/` (agentic, agentic_qa, memory_backends, arms subdirectories)

These are untracked and regenerated automatically by Python on import. The `.gitignore` already includes `__pycache__/`, so they will not be committed in the future.

---

## Updated .gitignore

The following patterns have been added to `.gitignore` to prevent future intermediate build artifacts:

```
# Additional LaTeX/Python build artifacts (Phase 6 cleanup)
*.xdv                  # XeLaTeX DVI output
*.bcf                  # BibLaTeX backend (.bcf)
*.run.xml              # BibLaTeX runtime
*.synctex.gz           # SyncTeX file (editor integration)
*.pyo                  # Compiled Python (optimization)
*.bak                  # Backup files
*.tmp                  # Temporary files
```

These join existing patterns for `*.aux`, `*.log`, `*.fls`, `*.fdb_latexmk`, `*.bbl`, `*.blg`, `__pycache__/`, `*.pyc`, `*.py[cod]`.

---

## Summary Table

| Category | Item | Decision | Reason |
|----------|------|----------|--------|
| **Untracked reports** | `t5_rows.csv`, `t5_stats.csv` | DELETE | Phase 4a drift invalid; superseded by Phase 5 agentic results |
| **Untracked reports** | `cd_diabetes_before.csv` | DELETE | Phase 4b exploratory; not in final evidence set |
| **Untracked reports** | `ontology_hpo_regression.log` | DELETE | Regression test, not part of pre-registered evidence |
| **Tracked figures (root)** | `fig*.{pdf,png}` | KEEP | Historical reference; not used by manuscript but tracked |
| **Tracked figures (individual/)** | TikZ outputs | KEEP FOR NOW | Design evolution history; consider archival in future cleanup |
| **Tracked figures (svg/)** | Figure 2, 5 + extended data | KEEP | Currently used by manuscript; Agent C owns extended data |
| **Untracked directories** | `ClaudeDesignBadFigures/` | ARCHIVE OR DELETE | Rejected design iterations; confirm with Agent A |
| **Untracked directories** | `References/` | KEEP IF ACTIVE, ELSE ARCHIVE | Design reference; keep if iteration ongoing |
| **Build artifacts** | `*.xdv`, `*.aux`, `*.log` (21 files, __pycache__ dirs) | REMOVED | Untracked intermediates; .gitignore updated |

---

**End of inventory.** Orchestrator to decide on deletions and archival.
