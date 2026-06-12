# Manuscript

## Target venue (decided 2026-06-12)

1. **Primary: NeurIPS 2026** (Datasets & Benchmarks track is the fallback
   track if the main track review profile looks unfavorable). Template:
   `neurips_2025.sty` (vendored here; swap in `neurips_2026.sty` when the
   2026 CFP publishes it — historically a near-identical revision).
   Submission is anonymous (`\usepackage{neurips_2025}` bare); the arXiv
   preprint uses the `[preprint]` option.
2. **Journal alternative: Nature Machine Intelligence.** NMI accepts
   free-format submissions ("format-neutral"); reformatting happens only at
   revision. So the NeurIPS-formatted PDF doubles as the NMI submission with
   a cover letter — no second template needed up front.
3. **Preprint: arXiv (cs.AI, cross-list cs.IR, cs.LG)** immediately after the
   confirmatory battery + internal review, using `[preprint]`.

## Rules

- No hand-typed numbers in the manuscript. Every table/figure is generated
  by `scripts/make_paper_assets.py` from CSVs under `reports/` (confirmatory
  runs only; exploratory numbers stay in docs/STATUS.md).
- Figure titles and filenames are unique (enforced by the asset script).
- Graphics tooling reference: `../GRAPHICS_STACK.md`.

## Build

```
cd paper/manuscript && latexmk -pdf main.tex
```
