# Paper Spine — The Committed Thesis

> Status: **committed** (2026-06-13). This is the single sentence the whole paper
> defends. Every experiment is either evidence for it or a registered falsifier of
> it. If a result contradicts this spine, we report the result and revise the
> spine — we do not bury it.

## 1. The thesis (one sentence)

**A structure-mapping memory (SMA) grounded in a *curated domain ontology* turns a
general LLM into a verifiable specialist: it retrieves by logical structure —
subsumption (is-a) plus higher-order relations (causes / part-of / regulates) —
that vector RAG and property-graph KGs discard, and so it beats both on the
rare / cross-vocabulary / high-stakes reasoning where hallucination is
unacceptable.**

Shorter, for talks: *"Give SMA the field's golden ontology and it out-retrieves
RAG and knowledge graphs on exactly the cases that matter — rare, novel, and
provenance-critical."*

## 2. Why structure beats vectors and property-graphs

Three retrieval families, three things they throw away:

| Family | What it indexes | What it discards | Failure mode |
|---|---|---|---|
| **Vector RAG** (dense / BM25 / hybrid) | surface-token similarity | logical relations; rarity is averaged out by frequency | retrieves the *frequent-and-similar*, misses the *rare-but-structurally-identical* |
| **Property-graph KG** (Neo4j, GraphRAG, HippoRAG) | entity nodes + typed edges, traversed by path | the *subsumption lattice* (what generalizes to what) and *higher-order* relations (relations over relations) | discriminates on entity arguments, not on the functor that carries the analogy; no admissible "how surprising is this match" weighting |
| **SMA + golden ontology** | functors over a constant subject + an is-a ascension lattice + higher-order statements | nothing the ontology encodes | (the bet) — falsifiers in §6 |

The mechanism that makes SMA different is the same one proven across Phase 3 and
4: **MAC/FAC discriminates on FUNCTOR identity, not entity arguments** (the 4b
"memory-as-functors" lesson), and the **surprisal scorer weights a matched
functor by −log₂ p ≈ its information content** — so a rare phenotype, a rare
attack technique, a rare failure mode counts for more, *by construction*, instead
of being averaged away. The ontology supplies the lattice for free: a specific
term ascends to a general one through the ontology's own is-a edges, with a
penalty ρ^distance — no bespoke relation engineering.

## 3. The generalizable recipe (the product idea)

From the flagship result (rare-disease, HPO — §5): SMA shines when you **mount a
mature domain ontology as the ascension lattice**. The recipe:

1. Encode each domain observation as a **FUNCTOR** over a constant subject node
   (content lives in functors, not entity args — the 4b lesson).
2. Plug the ontology's **is-a tree in as the lattice** (`canon.lattice.add(child,
   parent)`), so specific→general matching is automatic.
3. Lift the ontology's **typed relations** (part-of, causes, regulates,
   manifests, treats) into **higher-order statements** — relations over the
   functors — which is exactly the structure SME systematicity rewards.
4. Let the **surprisal scorer** do information-content weighting from the corpus.

The hand-built flat-tabular adapter FAILED in 4b precisely because flat tabular
has no ontology/structure to mount. So the rule for domain selection is: **prefer
domains with a mature ontology** (HPO, SNOMED, ICD, MONDO, ChEBI, GO; MITRE
ATT&CK; FIBO; CPC/IPC patents; legal taxonomies & citation graphs).

## 4. The four-tier adapter hierarchy (and the moat)

Adapters that map a domain artifact into SMA structure, ranked by trust:

| Tier | Source of structure | Trust | Governance |
|---|---|---|---|
| **Tier 0 — golden ontology** | a mature, community-curated ontology mounted directly (HPO, MITRE ATT&CK, OBO Foundry) | highest | frozen; versioned to the ontology release |
| **Tier 1 — expert-curated** | a domain expert hand-writes the encoder | high | frozen; ADR-007 sign-off |
| **Tier 2 — LLM-drafted + reviewed** | LLM proposes rules from train-only column summaries; admin reviews | medium | ADR-007: admin-only draft + sign-off; tainted until reviewed |
| **Tier 3 — LLM-drafted raw** | LLM proposes, no review | low | dev/experiments only; never production |

**The moat = Tier 0.** Anyone can call an LLM (Tier 2/3). The defensible asset is
a *registry of golden ontologies, each mounted correctly as an SMA lattice +
higher-order relations*, plus the router that picks the right one. Ontologies are
decades of curated expert labor (HPO, GO, MITRE) that competitors cannot cheaply
reproduce and that an LLM cannot hallucinate into existence. **Ontology-as-moat.**

Production default (ADR-007): **adapters are FROZEN.** Only admin users may
request LLM drafting (Tier 2) or give final sign-off. This keeps a deployed
system auditable.

## 5. Evidence so far

- **Rare-disease diagnosis (HPO + OMIM/Orphanet), exploratory, 2026-06-13.**
  Hard simulated patients (partial + imprecise presentation, 3 noise findings)
  vs ~2500 candidate diseases. SMA (HPO mounted as lattice, phenotype-as-functor,
  surprisal scorer) **ties** the SOTA-equivalent (Phenomizer IC best-match) on
  top-1 / MRR and **beats it on top-5 (+5pp) and top-10 (+6.5pp)** across 3 seeds.
  This is the spine's home turf: a real golden ontology, rarity-weighted, no
  bespoke engineering. → to be elevated to a **pre-registered paper arm**.
- **Phase 3 confirmatory** (separate, already frozen): cross-system transfer
  (vocab gap), rare-family retrieval, cite-or-abstain — the mechanisms the spine
  relies on, proven on the core benchmark battery.
- **Phase 4b honest null**: flat tabular (readmission, card fraud) → SMA reaches
  *statistical parity*, not a win, because the signal lives in fine-grained
  values, not structure. Reported as a **negative result that delimits the
  thesis**: no ontology ⇒ no SMA advantage. This is a feature — it tells you
  exactly when to reach for SMA.

## 6. Claims and registered falsifiers

The spine makes falsifiable claims. Each paper arm pre-registers the falsifier.

- **C1 (rarity):** on a golden-ontology domain, SMA beats vector RAG on top-k for
  k≥5 at rare/long-tail queries. *Falsifier:* paired-bootstrap CI for Δtop-5 vs
  the best baseline includes 0 after Holm correction.
- **C2 (cross-vocabulary):** SMA retrieves a structurally-identical case across a
  vocabulary gap that defeats lexical/dense retrieval. *Falsifier:* no
  significant transfer gain over hybrid RRF.
- **C3 (across fields):** the *same* mounting recipe wins in ≥2 unrelated
  golden-ontology domains (medicine + cyber). *Falsifier:* the second domain
  (MITRE ATT&CK) shows no SMA advantage → "across fields" claim is dropped.
- **C4 (no-ontology null):** on flat-tabular domains SMA does NOT beat value-based
  retrieval. *Already confirmed (4b)* — included as the boundary of the claim.

## 7. The universal adapter — what we build, and what we deliberately do NOT

**BUILD (correct, buildable, a moat):** a **universal OWL/OBO loader + ontology
registry + domain router.**
- *Loader*: parse any OWL (RDF/XML) or OBO file → a normalized `OntologyGraph`
  (terms, is-a edges, typed relations) → mount as a `Canonicalizer` lattice +
  higher-order-relation case builder. One code path, every ontology.
- *Registry*: named, versioned ontologies (`hpo@2024-... `, `attack@v15`), each
  with its mount config; pinned for reproducibility.
- *Router*: given a query/domain, pick the right ontology (or set) to retrieve
  against. Routing is a *selection* problem, not a merge.

**DO NOT build one merged omni-ontology graph.** Reasons, in order:
1. **Incompatible formalisms** — OWL-DL vs OBO vs SKOS vs property-graph schemas
   don't share semantics; naive union produces nonsense edges.
2. **Ontology alignment is unsolved** — cross-ontology term equivalence is an open
   research problem; a wrong `sameAs` silently corrupts every downstream match.
3. **Polysemy false-bridges** — "stress" (materials) ≠ "stress" (psychology);
   merging by label invents analogies that don't exist.
4. **Scale** — a single global lattice over all of OBO + FIBO + ATT&CK + CPC is
   neither tractable nor meaningful to ascend through.

**The principled rule: _merge WITHIN an aligned ecosystem, route ACROSS
ecosystems._** OBO Foundry ontologies already share upper-level scaffolding
(BFO + the Relation Ontology), so merging HPO+MONDO+GO+ChEBI *within* OBO is
sound and valuable. Crossing from OBO to FIBO to ATT&CK is a routing decision,
never a graph union.

## 8. Proof-domain roadmap

1. **HPO / rare disease** (medicine) — done exploratory, elevate to registered arm. ✅ evidence
2. **MITRE ATT&CK** (cyber) — second golden-ontology domain to earn the "across
   fields" claim (C3). Attack techniques as functors; the ATT&CK tactic/technique
   hierarchy as the lattice; `subtechnique-of` / `uses` as higher-order relations.
3. (stretch) **OBO-within-merge** demo — HPO+MONDO+GO mounted together via BFO/RO
   to show sound intra-ecosystem merge, contrasted with a routed cross-ecosystem
   query, operationalizing §7's rule.

## 9. Positioning vs the literature

- **vs Dense/Hybrid RAG**: we don't compete on frequent-similar recall; we win on
  rare/structural where embeddings collapse rarity.
- **vs GraphRAG / HippoRAG / KG-PPR**: they traverse entity graphs; we match on
  functors with an admissible rarity-weighted bound and an ascension lattice they
  don't have. (HippoRAG-2 is in our baseline battery.)
- **vs ontology-augmented LLMs / KBQA**: those inject ontology terms as *text* or
  *triples*; we use the ontology as the *retrieval geometry* itself.

---

*Companion docs:* `docs/Project_Architecture_Blueprint.md` (system),
`docs/ADR/007-dynamic-adapter-governance.md` (Tier 2/3 governance),
`configs/preregistration.md` (frozen dials & statistics). Memory:
`sma1-flagship-domains`, `sma1-release-and-governance`.
