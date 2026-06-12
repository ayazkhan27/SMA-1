# Figure 2 — The structure-mapping mechanism in depth

**Role:** the conceptual centerpiece — *why* SMA can match across systems that
share no words, and *what* it emits that RAG cannot. This is the figure that
carries the thesis. MAMMAL's Figure 2 plays the analogous "mechanism close-up"
role (their AF3 binding-pose comparison); ours is the cross-system alignment.

**Toolchain:** **TikZ/PGFPlots**, layouts pre-computed with **netgraph** or
NetworkX `dot`, rendered as vector. This figure must NOT be a Matplotlib
box-diagram — it needs precise edge bundling and nested geometry.

---

## Layout: two predicate DAGs facing each other, bridged three ways

**Left DAG — Base (stored BGL incident):** full cause-chain
`cause(timeout(R63-M0, treeLink), cause(retry(R63-M0, treeLink),
failure(R63-M0)))`. Nodes colored by role (slate entities, mid-teal first-order,
dark-teal higher-order). Depth rendered vertically (order axis on far left).

**Right DAG — Target (new Spirit session, the query):** the *same shape* with a
**disjoint vocabulary** — `sn-a12`, `fabric`, functors `q_timeout`/`q_retry`
drawn from a different lexicon. Visually identical skeleton, completely
different labels: the reader should see "same structure, different words" at a
glance.

### Bridge 1 — correspondences (the match hypotheses)
Double-headed dotted threads connecting corresponding nodes, **bundled** (not a
spaghetti of crossing lines — use TikZ `to[bend]` with a shared control region)
so the one-to-one mapping reads cleanly. Label the bundle "structurally
consistent gmap (injective, parallel-connected)."

### Bridge 2 — lattice ascension (the cross-vocabulary mechanism)
Between two non-identical corresponding functors, a **dotted violet** detour
*upward* to a shared ontology concept node (e.g. `timeoutEvent`), with the
penalty `ρ^dist` annotated on the climb. A small inset shows the predicate
lattice L as a tiny Hasse diagram with the ascension path lit. This is the
visual answer to "how do disjoint vocabularies match?" — they meet at a
declared ancestor, at a cost.

### Bridge 3 — trickle-down score (systematicity as arithmetic)
Along the right DAG, render the SES accumulation: each node carries a small
horizontal bar whose length = its trickle-down score `s(h) = σ₀·asc(h) + γ·Σ
parents`. Bars grow toward the deep `cause` root → the figure *shows* that deep,
interconnected matches dominate flat ones. A compact equation callout ties the
bars to the formula.

### The payload — candidate inference
Below the target DAG, the base's `failure(R63-M0)` is **projected** into the
target as `failure(sn-a12)`: a **red dashed** node with a dashed projection
arrow, tagged `status: hypothetical · verify or abstain`, and a provenance chip
`{base_id, gmap_id, SESₙ, support, skolems}`. Annotation: "the memory's output
is a checkable object with provenance — not generated text. No RAG pipeline can
emit this."

## Bottom strip — the contrast band
A thin three-cell strip showing what each paradigm retrieves for this query:
- **BM25/dense:** would rank the *surface distractor* (same words, broken
  structure) — small red ✗.
- **KG-PPR:** partial — entity overlap but no relational projection.
- **SMA:** the structural analog + the projected inference — teal ✓.
Mirrors MAMMAL Fig 2's binder/non-binder discriminative contrast.

---

**Data sources:** schematic, but the example case pair should be a *real*
retrieved pair from the confirmatory BGL→Spirit run (cite the actual case_ids in
the caption for honesty). SES bar values computed by `match_cases` on that pair.
**Caption skeleton:** "Figure 2. A single structure mapping across systems with
zero lexical overlap. Corresponding statements (dotted) form an injective,
parallel-connected gmap; non-identical functors match by bounded ascension
through the predicate lattice (violet); trickle-down scoring (bars) rewards
deep relational systems; and the unmatched base consequence is projected as a
provenance-tagged candidate inference (red, dashed) — the auditable output that
distinguishes analogical memory from generative recall."
