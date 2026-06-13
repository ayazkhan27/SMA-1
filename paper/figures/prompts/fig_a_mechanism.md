# Figure 1a — SMA mechanism

> Assumes the **SHARED BRIEF** you already have (science context + design directives + style/palette). This file is ONLY this figure's content — do not re-derive the shared rules.

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
