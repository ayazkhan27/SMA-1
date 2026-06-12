# Paper graphics & manuscript stack (reference)

Owner-supplied stack to draw on when building the manuscript, preprint, and
camera-ready assets. Keep this list in mind for every figure/table decision.

## Plotting
- Matplotlib (base; all paper figures)
- SciencePlots (journal styles — already in use via `scripts/make_paper_assets.py`)
- CMasher (perceptually uniform colormaps for heatmaps/score grids)
- adjustText (de-overlapping point labels on scatter plots)
- matplotlib-label-lines (inline line labels instead of legends)
- brokenaxes (axis breaks, e.g. SMA-vs-baseline gaps that dwarf the scale)

## Diagrams / flowcharts
- Mermaid (already: `paper/diagrams/*.mmd` — architecture, draft-adapter loop, tiered retrieval)
- Graphviz, diagrams (python), TikZ/PGFPlots (camera-ready vector diagrams in LaTeX)

## Graph visualization (for SME mapping/lattice figures)
- NetworkX, rustworkx, netgraph, pyvis, HyperNetX

## Tables
- pandas Styler / to_latex (booktabs), great-tables, Jinja2 templating

## ML comparison statistics
- scikit-posthocs, autorank (complement our sma/eval/stats.py: paired
  bootstrap + Holm-Bonferroni + Cliff's delta are pre-registered; autorank /
  critical-difference diagrams are presentation-layer extras)

## Interactive exploration (not for the paper PDF)
- plotnine, Plotly, pyvis

## Conventions already in force
- Figures land in `paper/figures/` as PDF+PNG, version-stamped, generated
  only by `scripts/make_paper_assets.py` (single source of truth).
- Figure titles and filenames must be unique (enforced in the script).
- Pre-verification numbers are hatched; un-hatch only from confirmatory runs.
- Target venue/template: see `paper/manuscript/README.md`.
