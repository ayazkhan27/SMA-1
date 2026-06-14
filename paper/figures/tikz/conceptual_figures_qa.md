# SMA-1 conceptual figures — QA note (tikz-diagrams skill)

Mode: research. Engine: xelatex (Arimo / Arial-metric, IBM-Plex-Mono fallback = \ttfamily).
Conceptual/schematic only — NO numbers, accuracies, bars, radars. Manuscript supplies all captions;
no caption prose is baked into any figure.

## fig1_overview.{tex,pdf,png} — 4-panel system overview (flagship)
- (a) Mount a curated ontology as retrieval geometry: patient + phenotype card -> 3-level HPO
  is-a lattice (root / abnormal nervous-system physiology / Seizure·Ataxia·Intellectual disability),
  solid is-a edges (arrow toward general), dashed part-of typed-relation edge, amber information-content
  glyph on the rare decisive term (Ataxia), violet ascension arrow, and the encoded analogical case
  with functor statements f_T(x) in teal, dashed correspondences back to matched concepts.
- (b) What each retriever indexes/discards: 3 lanes (Vector RAG token-cloud->one vector, rarity faded;
  Knowledge graph adjacency hops; SMA functors on is-a lattice with the rare node lit) each with
  green keep / red discard lists.
- (c) Structure-mapping retrieval: MAC funnel -> FAC two relational structures with dashed
  correspondences -> cite (green receipt/seal) / abstain (amber gauge) / novelty (red fork).
- (d) One universal loader; route across domains: loader+router hub with concentric ring, six labelled
  domain spokes (medicine/HPO, genomics/GO, cyber/ATT&CK, legal/CPC, finance/FIBO, chemistry/ChEBI),
  ontology NAMES only — no term counts, no radar, no results.
- Visual QA: zero text/element overlap; only QA false-positives are multi-panel "title_band" warnings
  (per-panel headers legitimately sit at panel top) and one rotated-label "crowding" warn in clear
  whitespace (ascension label vs. f_Seiz). Complexity outcome: KEEP.

## figM_architecture.{tex,pdf,png} — two-stage pipeline
- Stage 1 LOAD & REGISTER: format chips (.obo/.owl/.stix/.cpc) -> universal loader -> normalized
  ontology graph (is-a glyph) -> mount (is-a ascension lattice + case builder) -> registry/domain
  router -> case store cylinder.
- Stage 2 MATCH & DECIDE: encode (case = {f_T(x)}) -> MAC (admissible content bound) -> FAC
  (best-first SME alignment) -> surprisal scorer (−log2 p, rarity-weighted) -> {structural citation,
  calibrated abstention, SAGE novelty}. Frozen dials annotated lightly (max · rho=0.95 · delta=2).
  Dashed governance links (router selects domain memory; mounted geometry single source of truth).
- Visual QA: no overlap after lifting the is-a glyph above its box. Complexity outcome: KEEP.

## graphical_abstract.{tex,pdf,png} — single panel, column-friendly
- generalist LLM (fluent wavy glyph + red "unreliable on the rare tail" chip) + SMA memory
  (is-a lattice with lit rare node, structure-mapping over a curated ontology) => reliable,
  attributable specialist (green cites / amber abstains / red flags-novelty chips). One qualitative
  mechanism strip at the bottom (no metrics). Complexity outcome: KEEP.

Math/logic review: schematic (is-a arrows point specific->general; rarity = high −log2 p on the rare
term; MAC narrows then FAC aligns then scorer routes). No exact numeric claims.

Known TikZ gotcha fixed across all three: `color=NAME` placed AFTER `fill=NAME!pct` on a node overrides
the fill to a solid colour — use `text=NAME` for node text colour instead. Also avoided the reserved
TikZ key name `cap` (renamed to `hcap`).
