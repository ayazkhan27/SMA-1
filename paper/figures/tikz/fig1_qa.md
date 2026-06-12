# Figure 1 (TikZ) — QA log

- Engine: xelatex + fontspec (Arial). Source: scripts/figure1_tikz.py -> paper/figures/tikz/fig1.tex
- Static check (check_tikz_safety.py): PASS
- Visual QA (check_tikz_visual.py, research mode):
  - Resolved: panel D ribbon/label overlap (ribbon moved above rotated labels;
    method labels shortened); panel E legend/annotation overlap (legend moved to
    bottom row); panel C title vs panel B legend (row 1 lifted, B legend raised).
  - Remaining flags are title_band_collision false positives: the checker assumes
    ONE global title band, but this is a multi-panel composite where each panel
    carries its own gray-blue title. Manually inspected — accepted.
- Design critique gate: KEEP. One grammar per panel; economical labels; data
  panel (E) reads only reports/confirmatory CSVs.
- Math: panel-C formula in CM math (conventional); panel-E delta numerals in Arial.
