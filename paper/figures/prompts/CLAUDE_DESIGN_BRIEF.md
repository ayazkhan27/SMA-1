# Claude Design — Brief for the SMA-1 conceptual figures (Nature Machine Intelligence)

You are designing the **qualitative / conceptual** figures for a Nature Machine
Intelligence paper. **You do NOT make data figures** — every chart, bar, radar,
violin, histogram, and number is produced separately from the local results.
Your job is the *mechanism and architecture*: schematics that read as quiet,
rigorous scientific evidence. Do not invent or plot any number.

---

## 0. The hard boundary (read first)
- **You make:** Figure 1 (system overview), the architecture/pipeline schematic,
  the graphical abstract, and (optional) a conceptual "three capabilities" panel.
- **You do NOT make / must NOT include:** any measured value, accuracy, AUROC,
  bar, radar, or results panel. If a panel seems to want a number, omit it or
  leave a labelled slot. The previous version stuffed a results radar into the
  overview — **remove that idea entirely**; results live in separate data figures.

## 1. What SMA-1 is (so your metaphors are TRUE, not decorative)
SMA (Structure-Mapping Agentic Memory) is a **retrieval memory** that grounds a
language model — it is **not** a RAG/embedding system and **not** a neural model.

- It encodes each observation as a **functor over a subject**: a present term `T`
  on subject `x` becomes the statement `f_T(x)`; a typed relation `(s, r, o)`
  becomes the higher-order statement `r(f_s(x), f_o(x))`. A *case* is the set of
  such statements. ("Functor" = a predicate/term-constructor symbol — NOT a
  category-theory functor.)
- It mounts a **curated, expert-maintained ontology as retrieval geometry**: the
  ontology's **is-a hierarchy becomes an ascension lattice** along which a
  specific term matches a more general one (with a distance penalty `ρ^dist`),
  and an **information-content weight `−log₂ p`** makes the rare, decisive term
  dominate the match.
- Retrieval is **analogical structure-mapping** (the cognitive-science SME +
  MAC/FAC pipeline): a certified **MAC** admissible-content bound shortlists
  candidates, then **FAC** best-first alignment maps corresponding structure.
- A single **universal loader** ingests any open OBO/OWL ontology with no
  per-domain retrieval code; a **registry + domain router** select across domains
  (medicine, genomics, cyber, legal, finance, chemistry).
- Because retrieval is a structural alignment, every answer carries a **structural
  citation**, the agent can **abstain** when nothing structurally grounds the
  case, and it can **flag novelty** (a case matching nothing in memory is an
  *expectation violation*).

**The honest one-line thesis (anchor your visuals to this):** *a structure-mapping
memory over a curated ontology grounds a generalist LLM by the subsumption
hierarchy and rarity weighting that flat retrievers discard — and the structural
alignment itself is what survives when there is no surface signal at all.*

**What is load-bearing (so you emphasise the right things):** the **is-a
ascension lattice** and **information-content weighting** do the empirical lifting;
the **structural alignment** is what lets SMA solve cases with *zero word overlap*
that embeddings cannot. Do **not** over-emphasise "relations about relations" —
keep them present but secondary.

## 2. The figures you own

### Figure 1 — System overview (the flagship). Four panels, a–d.
- **(a) Mount a curated ontology as retrieval geometry.** raw artifact (e.g. a
  patient with phenotypes: *recurrent seizures, intellectual disability, ataxia*)
  → ontology (a small **is-a tree** glyph with one typed-relation edge, labelled
  HPO / GO / ATT&CK) → **analogical case** (a subject node bound to functor
  statements `f_T(x)`). Show the ascension arrow (specific → general).
- **(b) What each retriever indexes — and discards.** three colour-coded lanes:
  **Vector RAG** (a blurred token cloud → "collapses to one vector, discards
  rarity"); **Knowledge graph** (node–edge adjacency → "walks by adjacency, no
  subsumption"); **SMA** (functor nodes on an is-a lattice, the rare term
  highlighted → "keeps subsumption + rarity"). Use ✓keeps / ✗discards marks. No
  accuracy numbers.
- **(c) Structure-mapping retrieval with structural citation and abstention.**
  a left→right pipeline: **MAC** (a funnel: "admissible-bound shortlist") → **FAC**
  (DRAW two small relational structures with dashed lines aligning corresponding
  nodes — "best-first structural alignment") → **cite** (a receipt/seal icon —
  "structural provenance") → **abstain / flag novel** (a gauge or fork —
  "withholds when nothing grounds it; flags the unprecedented"). No numbers.
- **(d) One universal loader; route across domains.** a central **router** hub →
  spokes to domain glyphs (medicine cross, DNA helix, cyber shield, legal gavel,
  finance coin, chemistry flask), each labelled with its ontology name only (HPO,
  GO, ATT&CK, CPC, FIBO, ChEBI). **No term counts, no radar, no results.**

### Figure (Methods) — Architecture / pipeline schematic
The real architecture is a **retrieval pipeline**, not a multi-align neural net
(drop any "multi-align model" framing). Two stages, clean boxes + flow:
- **Adapter:** universal loader (OBO / OWL / STIX / CPC) → normalized ontology
  graph → **mount** (is-a lattice + higher-order case builder) → registry /
  domain router.
- **Matcher:** encode case → **MAC** (certified admissible content bound,
  inverted-index shortlist) → **FAC** (SME best-first alignment, systematicity) →
  **information-content / surprisal scorer** → **{structural citation, calibrated
  abstention, SAGE novelty}**. Annotate the frozen dials lightly
  (surprisal · max · ρ=0.95 · δ=2) if it fits without clutter; otherwise omit.

### Graphical abstract (single panel)
Generalist LLM (*fluent, unreliable on the rare tail*) **+** SMA memory
(*structure-mapping over a curated ontology*) **→** a reliable, attributable
specialist that **cites, abstains, and flags novelty**. Iconographic, ≤ one
column wide. **No metrics footer.**

### (Optional) Figure — Three capabilities flat retrievers lack
Three columns, icon + one-line "why SMA can / why RAG can't", **no bars/numbers**:
**Cite** (each hit *is* a structural alignment → checkable provenance);
**Abstain** (no structural ground → withhold; cosine is always "pretty high");
**Flag novelty** (matches nothing in memory → expectation violation).

## 3. Quality bar (why v1 was rejected; what to fix)
- **Readability over density.** Every label legible at print size. If text doesn't
  fit, *simplify the content* — never shrink to sub-readable micro-text.
- **Zero overlap** between text and diagram elements; generous whitespace; clear
  panel separation; one idea per panel.
- **MaMMAL-level polish, clean and high-contrast** (see the exemplars in the
  repo's `References/` folder — dense, iconographic, colour-blocked, schematic —
  the *quality* bar, not the layout to copy).
- Professional vector iconography; consistent stroke weight; semantic colour use.

## 4. Palette (keep consistent across figures)
SMA teal `#2E8AA6`, deep teal `#2E6B86`, first-order `#5AA9C4`, lattice violet
`#9B86C4`, amber `#D98A3D`, KG gold `#E7B15A`, grays `#C7CCD1 / #A7AFB6 / #7E8893`.
Domain accents: medicine `#C0504D`, genomics `#3E8E5A`, cyber `#4A6F8A`, legal
`#7A5DA8`, finance `#C68A2E`, chemistry `#3E8E5A`.

## 5. Real conceptual labels you may use (qualitative — safe, not data)
- **Medicine:** case *"recurrent seizures, intellectual disability, ataxia"* →
  HPO terms (Seizure, Intellectual disability, Ataxia) → a disease; is-a example
  *"Seizure is-a abnormal nervous-system physiology."*
- **Cyber:** a threat group's techniques → ATT&CK techniques; higher-order
  *"technique uses sub-technique."*
- **Domains → ontology names** (names only): medicine = HPO/MONDO/GO/Uberon;
  genomics = GO/ChEBI; cyber = ATT&CK/CAPEC/CWE; legal = CPC; finance = FIBO/US-GAAP.

## 6. Fonts
**Arimo** (metric-identical open Arial clone) — keep it. Source Serif 4 for caption
prose, IBM Plex Mono for IDs/code. No licensed faces to wire in.

---

*Deliver editable vector source (SVG/Figma) so panels can be refined; the
quantitative figures are produced separately and composited alongside yours.*
