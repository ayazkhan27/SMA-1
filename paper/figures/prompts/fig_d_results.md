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
**The evidence.** Across domains, SMA (teal) beats the **best enterprise RAG**
(gray) on the rare/tail slice, and provides trust capabilities the baselines score
**zero** on. The reader should grasp: "consistent wins where it matters, plus
capabilities the baselines structurally cannot match." This is a **data chart** —
render the numbers exactly; do not invent.

---

# FIGURE D — Results (DATA figure) · artboard 1500 × 520

Title top-left: **"d  SMA beats enterprise RAG on the rare/tail slice".**

**Main panel: grouped vertical bar chart.** x-axis = five domains
(Medicine, Genomics, Finance, Cyber, **Legal**); y-axis "tail top-5 accuracy (rare
slice)" 0–1.0. Two bars per domain: **SMA (teal `#2E8AA6`)** and **best enterprise
RAG (gray `#8A929B`)**. Values:
- Medicine: SMA 0.949, RAG 0.606  (Δ +0.33)
- Genomics: SMA 0.849, RAG 0.703  (Δ +0.16)
- Finance:  SMA 0.418, RAG 0.231  (Δ +0.17)
- Cyber:    SMA 0.766, RAG 0.749  (Δ +0.07)
- Legal:    **[PENDING — hatched placeholder bar pair labeled "running"]**
Print each Δ in small teal text above the SMA bar; thin significance brackets
"p<0.05". Clean axes, no top/right spines, no gridlines except a faint dashed
y=0.5 line. (Higher is better; SMA is the taller bar in every domain.)

**Right side-panel "Capabilities RAG cannot provide":** three stacked rows, icon +
one line each: "✓ Provenance — structural citation receipt"; "◑ Calibrated
abstention — risk-coverage AURC ≈0.02–0.11 vs 0.26–0.44"; "⚑ Novelty F1 ≈ 0.18
(every pure-RAG baseline = 0.00)". SMA values in teal, RAG values in gray.

(We may instead render this in matplotlib for numeric exactness; if you make it,
hold these numbers literally.)
