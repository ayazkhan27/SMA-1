## STYLE BLOCK (shared across all SMA figures)

**Output:** one self-contained, clean **SVG** file, vector, **transparent
background**, flat minimalist Nature/Science journal aesthetic — no 3-D, no heavy
gradients, no drop shadows beyond a very subtle 1px soft shadow. Keep all text as
**live editable text** (not outlined).
**Typography:** sans-serif (Helvetica/Arial). Panel title 18px bold slate `#5F6B78`;
node/box labels 13px `#2B333B`; small annotations 11px italic `#5F6B78`.
**Strokes:** 1.5px; arrowheads = small filled triangles (6px). Rounded rectangles
use a **6px corner radius**.
**Semantic palette — use these exact hexes:** SMA teal `#2E8AA6`; deep teal /
higher-order relations `#2E6B86`; first-order relations `#5AA9C4`; light-teal fill
`#E3EEF2`. Entity nodes slate `#7B8794`. is-a lattice violet `#9B86C4`, lavender
fill `#ECE8F4`. Candidate / abstain amber `#D98A3D`, amber fill `#FBF0E2`.
Vector-RAG / KG neutral gray `#8A929B`, KG gold `#E7B15A`. Green fill `#E6F0E6`;
grey fill `#F1F3F5`. Title/ink `#5F6B78`. Domain accents: medicine `#C0504D`,
genomics `#3E8E5A`, cyber `#4A6F8A`, legal `#7A5DA8`, finance `#C68A2E`.

---
# FIGURE D — Results (DATA figure — render exactly these numbers, do not invent) · artboard 1500 × 520

Title top-left: **"d  SMA beats enterprise RAG on the rare/tail slice".**

**Main panel: grouped vertical bar chart.** x-axis = five domains
(Medicine, Genomics, Finance, Cyber, **Legal**); y-axis "tail top-5 accuracy (rare
slice)" 0–1.0. Each domain has two bars: **SMA (teal `#2E8AA6`)** and **best
enterprise RAG (gray `#8A929B`)**. Values:
- Medicine: SMA 0.949, RAG 0.606  (Δ +0.33)
- Genomics: SMA 0.849, RAG 0.703  (Δ +0.16)
- Finance:  SMA 0.418, RAG 0.231  (Δ +0.17)
- Cyber:    SMA 0.766, RAG 0.749  (Δ +0.07)
- Legal:    **[PENDING — hatched placeholder bar pair labeled "running"]**
Print each Δ in small teal text above the SMA bar; thin significance brackets
"p<0.05". Clean axes, no top/right spines, no gridlines except a faint dashed
y=0.5 line.

**Right side-panel "Capabilities RAG cannot provide":** three stacked rows, icon +
one line each: "✓ Provenance — structural citation receipt"; "◑ Calibrated
abstention — risk-coverage AURC ≈0.02–0.11 vs 0.26–0.44"; "⚑ Novelty F1 ≈ 0.18
(every pure-RAG baseline = 0.00)". SMA values in teal, RAG values in gray.

NOTE: this is a data chart — we may instead render it in matplotlib for exactness;
if Claude Design makes it, hold these numbers literally.
