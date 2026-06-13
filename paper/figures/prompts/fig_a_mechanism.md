# Brief for the designer — please read first

## What you are illustrating (plain-language science brief)

We are a machine-learning research group writing a paper for *Nature Machine
Intelligence*. You are designing a figure for it. Here is the science, so every
shape and label you draw is meaningful.

**The problem.** Large language models (LLMs, like ChatGPT) are fluent but
unreliable *specialists* — in medicine, law, security, and finance they give
confident answers that are sometimes wrong or fabricated. The standard fix is
**Retrieval-Augmented Generation (RAG)**: before answering, the system looks up
relevant knowledge. Mainstream RAG retrieves by **embedding similarity** — it turns
text into a point in a high-dimensional vector space and grabs the nearest points.
This works for common, look-alike queries but fails on the **rare, structurally
defined cases that define expertise**, and it cannot explain *why* it retrieved
something or tell when it is unsure.

**Our method — SMA = Structure-Mapping Agentic Memory.** Instead of a vector, SMA
represents each piece of knowledge as a **structure**: a small graph of objects and
the *labelled relationships* between them. It retrieves by **matching that
structure** (analogy) — the way a clinician recognises that a new patient is "like"
a known disease. SMA is grounded in a **golden ontology**: an expert-curated map of
a field's concepts and how they relate (e.g. the Human Phenotype Ontology in
medicine). Because it matches on **logical structure** — including "is-a"
hierarchies and *relations between relations* — and weights **rare features** more
heavily, it wins exactly where vector RAG fails (the rare "tail"), and it can do
three things RAG architecturally cannot: **cite** the reason for a match
(provenance), **abstain** when nothing matches, and **flag novel** inputs it has
never seen.

**What we test.** We hold one AI agent fixed and **swap only its retrieval memory**
— SMA versus the best enterprise RAG and knowledge-graph systems — across **five
unrelated fields** (medicine, genomics, finance, cyber, legal) and measure accuracy
on the **rare/tail slice** plus those three trust capabilities.

## Vocabulary you'll see in labels (so they read correctly)
- **Ontology** — an expert-curated graph of a field's concepts + relationships.
- **is-a lattice / subsumption** — a hierarchy ("seizure is-a neurological sign");
  lets a specific term match a more general one.
- **Higher-order relation** — a relationship whose endpoints are *themselves*
  relationships (the thing vector embeddings cannot represent).
- **Vector RAG** — mainstream baseline: text → a single point in a fuzzy cloud.
- **Knowledge graph (KG)** — entities joined by edges (a competing baseline).
- **MAC / FAC** — SMA's two-stage retrieval: a fast filter, then careful alignment.
- **Cite / abstain / novelty** — the trust capabilities RAG lacks.

## Journal standard — please design to this
For *Nature Machine Intelligence*: **flat vector, multi-panel, lettered (a, b, c…),
restrained and colour-blind-safe, sans-serif, information-dense but uncluttered**,
with generous whitespace. A good figure **communicates a mechanism at a glance** —
every shape and arrow carries meaning, nothing is decorative; long explanation goes
in the caption, not the artwork. Target a crisp print size (~180 mm wide).

---
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

## What this panel must convey
This is the conceptual heart of the paper. A reader should instantly grasp **how
SMA differs from vector RAG**: SMA *preserves and matches structure* (relationships,
hierarchy, the rare feature), whereas RAG *collapses everything into one point in a
fuzzy cloud* and so loses the rarity and the relations. By the end the reader thinks:
"SMA retrieves by logical structure — that's why it can also cite, abstain, and flag
novelty." Make the SMA pipeline dominant and the RAG inset clearly subordinate.

---

# FIGURE A — "What is SMA?" (the mechanism) · artboard 1600 × 760

Draw a **left-to-right pipeline in four stages** plus a contrast inset along the
bottom. Title top-left: **"a  Structure-mapping retrieval".**

**Stage 1 (x≈40–250): raw input.** Rounded rectangle (grey fill `#F1F3F5`, slate
border): bold header "Clinical case", then three lines: "recurrent seizures",
"intellectual disability", "ataxia".

**Arrow → "encode"** (teal `#5AA9C4`, horizontal, 11px italic label above).

**Stage 2 (x≈300–700): structured representation.** Central **slate circle
(radius 22)** labeled "subject". Three **first-order arrows** (teal `#5AA9C4`,
fanning at ≈ −25°, 0°, +25°) to **three circles (radius 15, light-teal fill)** =
phenotype terms; each arrow labeled "presents". Above the top phenotype, an **is-a
ascension lattice**: three small rounded rects stacked vertically and **narrowing
upward** (trapezoidal stack, violet `#9B86C4` outline, lavender fill), joined by
short upward arrows; label the upward direction "is-a ascension (ρ^d)" in violet
11px. Then **one higher-order relation**: a smooth **curved bracket (deep teal
`#2E6B86`) connecting two of the first-order arrows to each other** (the arrows, not
the circles), labeled "causes" — emphasising a *relation between relations*. Give
one phenotype circle a thicker teal ring + a tiny "rare" tag.

**Arrow → "structure-match (MAC → FAC)"** (teal `#2E8AA6`).

**Stage 3 (x≈760–1120): analogical memory.** Three **case-cards** with a slight 6°
fan; the **best-matching card highlighted** (teal `#2E8AA6` border, light-teal
fill), the other two muted grey. **3–4 dashed grey correspondence lines** link
stage-2 nodes to the matched card's nodes (the structural alignment).

**Stage 4 (x≈1180–1560): verifiable outputs.** Three stacked chips: green chip
`#E6F0E6` "✓ citation (provenance)"; an amber **semicircular gauge** `#FBF0E2`,
needle near full, "abstain when no match"; amber chip with a **flag glyph**
"⚑ novelty (new entity)".

**Bottom inset (full width, y≈600–740, thin grey rounded box):** header "How vector
RAG sees the same case". Left: the same three-symptom case; arrow "embed" → a
**dotted ellipse cloud of ~40 scattered small gray dots** with the case reduced to
**one solid gray dot** lost among them. Caption (red-grey italic): "structure and
rarity discarded — no relations, no subsumption, no rare-feature emphasis." Keep
the inset clearly subordinate (smaller) to the pipeline above.

---

## Designer Q&A — binding refinements (apply these)
1. **Single takeaway (build hierarchy around this):** "SMA matches the logical
   *structure* of a case (relations + hierarchy + rare feature) that vector RAG
   collapses into one point and loses." The dominant visual is the **structured-
   graph (SMA) vs point-in-cloud (RAG) contrast**; all else subordinate.
2. **Reusable system:** this is panel a of Figure 1, but build the structured-graph
   glyph, the RAG dot-cloud glyph, and the cite/abstain/novelty icons as **reusable
   components** (they recur in Figure 3).
3. **MAC → FAC = two light sub-steps:** MAC = a fast *filter funnel* (many → short
   list); FAC = *careful alignment* (the dashed correspondence lines) on the
   survivors. Labeled, modest size.
4. **Push the rare feature hard:** the rare phenotype is visibly heavier (thicker
   teal ring, slightly larger, a "−log p · high weight" tag); in the RAG inset show
   that SAME feature fading into the average ("rarity averaged away").
5. **No emoji — custom flat icons:** Cite = provenance *receipt* (doc + check + 2–3
   evidence lines); Abstain = *gauge* with needle below a threshold ("hold");
   Novelty = *starburst/flag* ("new — escalate"). One-word labels.
6. **RAG contrast = prominent counterpoint, visually subordinate:** own grey band,
   ~25–30% area, desaturated so SMA reads first; keep the "✗ cannot cite / abstain /
   flag novelty" punchline.
7. **Single worked example** (clinical/HPO); breadth lives in Figure 1c.
8. **Real ontology IDs on anchor nodes:** e.g. "Seizure (HP:0001250)" on the key /
   rare nodes; short names elsewhere.
9. **Typeface:** Helvetica/Arial, **embedded/outlined** (never a system serif/emoji
   fallback), final 5–7 pt.
10. **Palette:** keep the teal set but ensure colour-blind-safe (hue+lightness
   separation; never red-alone for "wrong" — always pair red with the ✗ shape).

## Designer Q&A — round 2 (additive sharpenings; round-1 choices unchanged)
1. **Show the ontology grounding:** the is-a lattice emerges from a small labelled
   source tab "HPO · expert-curated ontology" with a thin "mount" connector — the
   structure is grounded in expert curation, not invented.
2. **Abstain = the decision rule, visible:** gauge with a threshold tick τ, needle
   BELOW it, and the consequence "best alignment 0.31 < τ → abstain (no answer)".
3. **Novelty = show the trigger:** query fails to align with anything in the store
   (greyed candidates, no correspondence lines, "∅ no aligned case") → starburst
   "novel — escalate".
4. **Higher-order "causes":** keep it, but it must join two ARROWS (relations),
   not nodes — the "relation over relations" reading must be unambiguous.
5. **RAG inset shows FAILURE:** embed → nearest dot → "returns: <common disease>
   ✗ (wrong)" — confident & incorrect, not just a fuzzy cloud. Pair red with ✗.
6. **Cite = name case + features:** receipt "✓ matched Case #214 (rare disease X);
   justified by Seizure (HP:0001250), Ataxia (HP:0001251), …".
7. **No reversals** of round-1 choices.
8. **Reading-order scaffolding:** small numbered markers 1→2→3→4 (or a thin guide
   spine) so a non-specialist parses input→encode→match→output at a glance.
9. **Deliverable:** SVG (editable, embedded text) + high-res PDF; design at TRUE
   print size — full-width Nature figure ≈180 mm wide, 5–7 pt body text at that size.
