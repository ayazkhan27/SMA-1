# Figure 9 — SAGE consolidation: schemas emerge from experience

**Role:** shows the memory *learns structure without training* — generalizations
form by analogical assimilation, fact probabilities are frequencies, weak facts
wear away. Supports the "robust to unseen concepts" mechanism (a novel incident
matches a *schema* even when it matches no single exemplar).

**Toolchain:** **netgraph** / NetworkX for the pool topology, Matplotlib for the
probability dynamics; CMasher sequential for probability shading.

## Panel A — pool formation (graph)
A generalization pool drawn as a graph: exemplar cases (small DAG glyphs) cluster
into a **generalization** node (a schema, drawn as a translucent merged DAG);
two mutually-assimilable outliers visibly seed a new generalization. Edges
weighted by SESₙ; assimilation threshold θ_a annotated. Disjunctive concepts show
as multiple generalizations in one pool (number of clusters discovered, not set).

## Panel B — the schema, with fact probabilities
The emergent schema DAG with each fact shaded by its support frequency
(CMasher sequential); facts below θ_p drawn faded ("wearing away"). A reader sees
which relations are *core* to the schema vs incidental.

## Panel C — probability dynamics
Line: a fact's probability = n_f/N_g over constituents as the pool grows;
core facts rise toward 1, noise facts fall below θ_p and get pruned. Annotate
"probabilities are frequencies — nothing learned, nothing tuned beyond θ_p."

## Panel D — schema-recovery validation
Bar: on synthetic streams with known generating schemas, recovered-schema fact-set
F1 (gate G6 target ≥ 0.9), and "probabilities equal analytic frequencies exactly"
as an exactness check.

**Data sources:** `SagePool.stats()` over a synthetic stream; a dedicated
`reports/sage_recovery.csv` (Phase deliverable). 
**Caption point:** "Generalization pools discover schemas by analogical
assimilation (A,B); fact probabilities are pure support frequencies that
strengthen core structure and prune noise (C); recovered schemas match the
generators (D) — consolidation without gradient training."
