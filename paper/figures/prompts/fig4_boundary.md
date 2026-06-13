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
**Honesty / scope.** SMA's advantage is *specific to structure*. This schematic
contrasts two faces of the SAME domain — credit-card fraud — to define when SMA wins
and when it does not. Flat-tabular fraud (rows of numbers) has no structure to
exploit → SMA ties standard models (a reported *null*). Structural fraud (a
transaction/typology graph) → SMA wins and can flag a *novel* scheme. The reader
should grasp: "the 'tail' that SMA owns is the *structural/novel* tail, not mere
statistical rarity — and the authors say so plainly." This is the schematic panel of
Figure 4; the quantitative panels (de-risk parity-vs-oracle, relational-richness)
are separate data plots.

---

# FIGURE 4 (schematic panel) — When SMA helps: the structural boundary · artboard 1500 × 620

Title top-left: **"The advantage is structural, not statistical".** Two halves
separated by a vertical divider.

**LEFT half — "Flat-tabular fraud → no advantage (null)" (muted grey theme).** Show a
small spreadsheet/table glyph: a credit-card transaction as a row of ~6 numeric
cells ("amount, time, V1, V2, …"). Arrow → two equal-height bars labeled "SMA" and
"value-based model" at the same level, with a grey "=" between them and a caption
"signal lives in fine-grained values — structure-mapping has nothing to grab; SMA =
baseline (reported null)."

**RIGHT half — "Structural / typology fraud → SMA wins (amber/teal theme).** Show the
same fraud as a small **graph**: nodes "account → device → merchant → counterparty"
with labelled edges, tagged with a typology label (e.g. "synthetic-identity chain").
Arrow → SMA retrieves the matching rare typology (teal ✓) AND an amber "⚑ novel
typology — escalate" flag for an unseen pattern; a short caption "decisive signal is
a rare/novel *structure* — SMA's home; embeddings/classifiers miss it."

Between the halves, a small centered banner: **"define the tail = rare/novel
*structure*, not statistical rarity."** Keep both halves visually balanced; the left
deliberately flat/grey, the right relational/coloured.
