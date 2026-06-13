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
# FIGURE A — "What is SMA?" (the mechanism) · artboard 1600 × 760

Draw a **left-to-right pipeline in four stages** plus a contrast inset along the
bottom. Title top-left: **"a  Structure-mapping retrieval".**

**Stage 1 (x≈40–250): raw input.** A rounded rectangle (grey fill `#F1F3F5`, slate
border): bold header "Clinical case", then three lines: "recurrent seizures",
"intellectual disability", "ataxia".

**Arrow → "encode"** (teal `#5AA9C4`, horizontal, 11px italic label above).

**Stage 2 (x≈300–700): structured representation.** A central **slate circle
(radius 22)** labeled "subject". From it, **three first-order arrows** (teal
`#5AA9C4`, fanning at ≈ −25°, 0°, +25°) to **three circles (radius 15, light-teal
fill)** = phenotype terms; each arrow labeled "presents". Above the top phenotype,
an **is-a ascension lattice**: three small rounded rects stacked vertically and
**narrowing upward** (trapezoidal stack, violet `#9B86C4` outline, lavender fill),
joined by short upward arrows; label the upward direction "is-a ascension (ρ^d)" in
violet 11px. Then **one higher-order relation**: a smooth **curved bracket
(deep teal `#2E6B86`) connecting two of the first-order arrows to each other** (the
arrows, not the circles), labeled "causes" — emphasizing a *relation between
relations*. Give one phenotype circle a thicker teal ring + a tiny "rare" tag.

**Arrow → "structure-match (MAC → FAC)"** (teal `#2E8AA6`).

**Stage 3 (x≈760–1120): analogical memory.** Three **case-cards** stacked with a
slight 6° fan; the **best-matching card highlighted** (teal `#2E8AA6` border,
light-teal fill), the other two muted grey. Draw **3–4 dashed grey correspondence
lines** linking stage-2 nodes to the matched card's nodes (the structural
alignment).

**Stage 4 (x≈1180–1560): verifiable outputs.** Three stacked chips: green chip
`#E6F0E6` "✓ citation (provenance)"; an amber **semicircular gauge** `#FBF0E2`
needle near full, "abstain when no match"; amber chip with a **flag glyph**
"⚑ novelty (new entity)".

**Bottom inset (full width, y≈600–740, thin grey rounded box):** header "How vector
RAG sees the same case". Left: the same three-symptom case; arrow "embed" → a
**dotted ellipse cloud of ~40 scattered small gray dots** with the case reduced to
**one solid gray dot** lost among them. Caption (red-grey italic): "structure and
rarity discarded — no relations, no subsumption, no rare-feature emphasis." Keep
this inset clearly subordinate (smaller) to the pipeline above.
