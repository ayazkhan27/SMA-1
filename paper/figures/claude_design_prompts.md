# Claude Design prompts — SMA paper figures (individual, SVG)

Each figure is a **standalone SVG** (we stitch later). Paste one prompt at a time.
A shared **STYLE BLOCK** precedes all of them — prepend it to every prompt so the
figures share one visual language.

---

## STYLE BLOCK (prepend to every figure prompt)

> **Output:** one self-contained, clean **SVG** file, vector, **transparent
> background**, flat minimalist Nature/Science journal aesthetic — no 3-D, no
> heavy gradients, no drop shadows beyond a very subtle 1px soft shadow.
> **Typography:** sans-serif (Helvetica/Arial). Panel title 18px bold in slate
> `#5F6B78`; node/box labels 13px `#2B333B`; small annotations 11px italic
> `#5F6B78`. **Strokes:** 1.5px; arrowheads = small filled triangles (6px).
> Rounded rectangles use a **6px corner radius**.
> **Semantic palette — use these exact hexes consistently:**
> SMA teal `#2E8AA6`; deep teal / higher-order relations `#2E6B86`; first-order
> relations `#5AA9C4`; light-teal fill `#E3EEF2`. Entity nodes slate `#7B8794`.
> is-a lattice violet `#9B86C4`, lavender fill `#ECE8F4`. Candidate/abstain amber
> `#D98A3D`, amber fill `#FBF0E2`. Vector-RAG / KG neutral gray `#8A929B`, KG gold
> `#E7B15A`. Green fill `#E6F0E6`; grey fill `#F1F3F5`. Title/ink `#5F6B78`.
> Domain accents: medicine `#C0504D`, genomics `#3E8E5A`, cyber `#4A6F8A`,
> legal `#7A5DA8`, finance `#C68A2E`.

---

## FIGURE A — "What is SMA?" (the mechanism)  ·  artboard 1600 × 760

> Draw a **left-to-right pipeline in four stages** plus a contrast inset along the
> bottom. Title top-left: **"a  Structure-mapping retrieval".**
>
> **Stage 1 (x≈40–250): raw input.** A rounded rectangle (grey fill `#F1F3F5`,
> slate border), label header "Clinical case" in bold, then three lines:
> "recurrent seizures", "intellectual disability", "ataxia".
>
> **Arrow → "encode"** (teal `#5AA9C4`, horizontal, label above in 11px italic).
>
> **Stage 2 (x≈300–700): structured representation.** A central **slate circle
> (radius 22)** labeled "subject". From it, **three first-order arrows** (teal
> `#5AA9C4`, fanning out at roughly −25°, 0°, +25°) to **three smaller circles
> (radius 15, light-teal fill)** = phenotype terms. Each arrow labeled "presents".
> Above the top phenotype circle, draw an **is-a ascension lattice**: three small
> rounded rectangles stacked vertically and **narrowing as they go up**
> (trapezoidal stack, violet `#9B86C4` outline, lavender fill), connected by short
> upward arrows; label the upward direction "is-a ascension (ρ^d)" in violet,
> 11px. Then draw **one higher-order relation**: a smooth **curved bracket
> (deep teal `#2E6B86`) that connects two of the first-order arrows to each other**
> (not the circles — the arrows), labeled "causes" — visually emphasizing that
> this is a *relation between relations*. Highlight one phenotype circle with a
> thicker teal ring and a tiny "rare" tag to show rarity-weighting.
>
> **Arrow → "structure-match (MAC → FAC)"** (teal `#2E8AA6`).
>
> **Stage 3 (x≈760–1120): analogical memory.** Three **case-cards** stacked with a
> slight 6° fan. The **best-matching card is highlighted** (teal `#2E8AA6` border,
> light-teal fill); the other two are muted grey. Draw **3–4 dashed grey
> correspondence lines** linking the query's nodes (stage 2) to the matched card's
> nodes, to show the structural alignment.
>
> **Stage 4 (x≈1180–1560): verifiable outputs.** Three small stacked chips:
> a green chip `#E6F0E6` "✓ citation (provenance)"; an amber **semicircular gauge**
> `#FBF0E2` needle near full, labeled "abstain when no match"; an amber chip with
> a **flag glyph** "⚑ novelty (new entity)".
>
> **Bottom inset (full width, y≈600–740, inside a thin grey rounded box):**
> header "How vector RAG sees the same case". On the left, the *same* three-symptom
> case; an arrow "embed" to the right into a **dotted ellipse cloud of ~40
> scattered small gray dots** with the case reduced to **one solid gray dot** lost
> among them. Caption in red-grey italic: "structure and rarity discarded — no
> relations, no subsumption, no rare-feature emphasis." This inset is the visual
> argument; keep it clearly subordinate (smaller) to the main pipeline above.

---

## FIGURE B — Universal adapter  ·  artboard 1300 × 620

> Title top-left: **"b  One universal loader, any ontology".** Left-to-right flow.
>
> **Left (x≈40–230): five source chips** stacked vertically (rounded rects,
> light-teal fill `#E3EEF2`, slate border), labeled top-to-bottom: "OBO",
> "OWL / TTL", "STIX 2.1", "CPC XML", "XBRL". A faint dashed grey bracket to their
> left labeled "ontology sources" (11px).
>
> **Five converging arrows** (deep teal `#2E6B86`) from each chip into a single
> **flat-top hexagon** (x≈430–640, amber fill `#FBF0E2`, 1.4px border) labeled
> "universal loader + mount" (two lines, centered).
>
> **Arrow →** a rounded rectangle (green fill `#E6F0E6`) labeled
> "normalized graph — is-a lattice + higher-order relations" (two lines).
>
> **Arrow →** a rounded rectangle (lavender fill `#ECE8F4`) labeled
> "registry + domain router".
>
> Below the flow, centered 11px italic caption: "any ontology → one structure-
> mapping retrieval geometry (merge within an ecosystem, route across)."

---

## FIGURE C — Five golden-ontology domains  ·  artboard 1500 × 420

> Title top-left: **"c  Tested across five domains".** A **horizontal row of five
> shield badges** (rounded-top pentagon "shield" shape, ~230px tall, ~250px wide,
> evenly spaced). Each shield: a **field icon** at top (simple line glyph), the
> **domain name** in bold (in that domain's accent color), and below it the
> **ontology + scale** in 11px. A thin grey arrow enters each shield from a small
> shared "domain router" pill centered above the row.
>
> 1. **Medicine** — accent `#C0504D`, icon = medical cross / caduceus —
>    "HPO · MONDO · GO · Uberon".
> 2. **Genomics** — accent `#3E8E5A`, icon = DNA double helix — "GO + GOA".
> 3. **Cyber** — accent `#4A6F8A`, icon = shield with keyhole — "ATT&CK · CAPEC · CWE".
> 4. **Legal / IP** — accent `#7A5DA8`, icon = balance scales — "CPC (254k nodes)".
> 5. **Finance** — accent `#C68A2E`, icon = coin / dollar — "US-GAAP (FIBO routed)".
>
> Keep icons flat single-color line art in each shield's accent color.

---

## FIGURE D — Results (data figure)  ·  artboard 1500 × 520

> NOTE: this is a **data chart** — render exactly these numbers; do not invent.
> Title top-left: **"d  SMA beats enterprise RAG on the rare/tail slice".**
>
> **Main panel: grouped vertical bar chart**, x-axis = five domains
> (Medicine, Genomics, Finance, Cyber, **Legal**), y-axis = "tail top-5 accuracy
> (rare slice)" from 0 to 1.0. For each domain two bars: **SMA (teal `#2E8AA6`)**
> and **best enterprise RAG (gray `#8A929B`)**. Values:
> - Medicine: SMA 0.949, RAG 0.606
> - Genomics: SMA 0.849, RAG 0.703
> - Finance: SMA 0.418, RAG 0.231
> - Cyber: SMA 0.766, RAG 0.749
> - Legal: **[PENDING — leave a hatched placeholder bar pair, label "running"]**
> Above each SMA bar print the delta in small teal text: "+0.33", "+0.16",
> "+0.17", "+0.07", "[tbd]". Thin black significance brackets with "p<0.05".
> Clean axis, no top/right spines, no gridlines except a faint y=0.5 dashed line.
>
> **Right side-panel "Capabilities RAG cannot provide":** three stacked rows, each
> an icon + one line: "✓ Provenance — structural citation receipt";
> "◑ Calibrated abstention — risk-coverage AURC ~0.02–0.11 vs 0.26–0.44";
> "⚑ Novelty F1 ≈ 0.18 (every pure-RAG baseline = 0.00)". Use teal for the SMA
> values, gray for the RAG values.

---

### Handoff notes
- Ask Claude Design for **each figure as its own `.svg`**, transparent background,
  fonts as live text (not outlined) so we can edit labels.
- Figure D's numbers are final for 4 domains; legal fills in when its run lands.
- We compose A–D into the Z-flow 2×2 master afterward (or keep them as separate
  display items if that reads better).
