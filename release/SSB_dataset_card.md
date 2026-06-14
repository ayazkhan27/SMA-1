---
license: apache-2.0
task_categories:
  - feature-extraction
  - question-answering
tags:
  - structure-mapping
  - analogical-retrieval
  - zero-lexical-overlap
  - predicate-lattice
  - retrieval-benchmark
  - rare-events
  - reasoning
  - sma
pretty_name: Structure Synthesis Benchmark (SSB)
---

# Structure Synthesis Benchmark (SSB)

## Dataset summary

The **Structure Synthesis Benchmark (SSB)** is a zero-lexical-overlap structural
retrieval benchmark designed to test whether a retrieval system matches on
**relational structure alone**, with no surface vocabulary shared between the
query and the target analog.

SSB is part of the SMA-1 evaluation suite (paper: *"Structure-Mapping Agentic
Memory"*, Ayaz Khan, 2026, under review at *Nature Machine Intelligence*).

**Headline result (from `reports/confirmatory/ssb_summary.csv`, seeds 41 and
43, n = 100 per seed):**

| Method | Forced-choice r@1 | Library r@1 |
|---|---|---|
| SMA (structure-mapping) | **1.000** (SD 0.000) | **0.895** (SD 0.007) |
| BM25 | 0.000 (SD 0.000) | 0.000 (SD 0.000) |
| TF-IDF Dense | 0.000 (SD 0.000) | 0.000 (SD 0.000) |

Cliff's δ = 0.895, p_Holm = 0.0004 (paired bootstrap, 10 000 resamples).

---

## Dataset construction

### Core principle: disjoint per-triple vocabularies

Each benchmark triple (base, target, distractor) uses **independently sampled
random functor names** so that the query and its true analog share **zero surface
vocabulary**.  The only bridge is a **declared predicate lattice**: the
query functor and the analog functor are mapped to the same abstract concept
in the lattice, and retrieval requires ascending at most δ = 2 hops
(penalised by ρ^distance = 0.95^hops).

This construction directly operationalises Gentner's systematicity principle:
a system that matches on surface co-occurrence will score every candidate
equally (zero lexical signal); only a system that ascends the predicate lattice
to find structural correspondences will score the true analog higher.

### Previous (broken) construction — what we fixed

Prior versions used a `far_` / `near_` prefix naming trick: the same functor
vocabulary was reused across triples, and the "far" analog was simply the
renamed version.  This was circular — the benchmark unknowingly told the system
which triple was the analog.  The current generator eliminates this entirely:

- Each triple receives a fresh randomly-sampled vocabulary (UUID-derived functor
  names with no shared substrings).
- Distractors are star-rewired (provably non-isomorphic to chain-structured
  analogs for width ≥ 3) so a correct matcher cannot accidentally score a
  distractor 1.0.
- The MAC screening stage uses the ancestor-closure feature vector (blueprint
  §2.7) so lattice-bridged vocabularies intersect at screening; the Lemma-2
  bound remains admissible with ancestor features.

### Generation parameters (frozen at prereg-v1)

| Parameter | Value |
|---|---|
| Lattice ascension depth δ | 2 hops |
| Ascension penalty ρ | 0.95 per hop |
| Structural width | ≥ 3 (chain structure) |
| Vocabulary collision probability | < 10⁻⁶ (UUID-space) |
| Seeds used in confirmatory runs | 41, 43 (n = 100 each) |
| Seeds used in development / calibration | 29, 31 |

---

## Dataset splits

| Split | Purpose | Seeds | n triples | Status |
|---|---|---|---|---|
| `dev` | Calibration (rho dial) | 29, 31 | 200 | Released |
| `test` | Confirmatory evaluation | 41, 43 | 200 | Released |
| `forced_choice` | One analog + one distractor per query | all | see above | Released |
| `library` | Query against a pool of 24 indexed cases | all | see above | Released |

The **forced-choice** split presents each query with exactly one true analog
and one star-rewired distractor; r@1 = 1 means the system always ranks the
analog first.

The **library** split presents each query against a 24-case indexed library;
r@1 measures whether the structural match tops the library ranking.

---

## Intended use

SSB is intended as a **diagnostic test** for retrieval systems that claim to
match on relational structure rather than surface similarity:

1. **Structural retrieval systems** (structure-mapping engines, SME-style,
   predicate-lattice-based) — the benchmark was designed to reward these.
2. **Ablation studies** — a system that degrades from structural to surface
   retrieval should drop from r@1 ≈ 1.0 to r@1 ≈ 0.0 on SSB.
3. **Unit testing** — the forced-choice split is included in the SMA-1 test
   suite as gate G4 (`test_macfac_lattice_bridges_disjoint_vocabularies`).

SSB is **not** intended as a benchmark for:
- General-purpose information retrieval (no natural language; no text documents)
- Semantic similarity (there is no semantic content — functors are arbitrary symbols)
- Knowledge graph completion (no real-world entities or relations)

---

## Limitations and caveats

1. **Synthetic by construction.** SSB uses randomly generated functor names and
   artificially constructed predicate lattices.  Performance on SSB does not
   directly predict performance on real-domain retrieval tasks (see the 5-domain
   agentic suite in the SMA-1 paper for real-domain evaluation).

2. **The lattice is declared, not learned.** The predicate lattice used by SMA
   is supplied at evaluation time.  A system without a compatible lattice (e.g.
   pure BM25, dense RAG) will score 0.000 by construction — the benchmark is a
   test of whether the mechanism exists, not how well it generalises to unseen
   lattices.

3. **Chain structure only.** The current generator uses chain-shaped base cases
   (linear predicate chains).  Richer topologies (trees, DAGs) are not yet in
   the benchmark.

4. **Small n.** Confirmatory evaluation uses n = 100 per seed × 2 seeds = 200
   triples.  Statistical power is sufficient for the observed effect sizes
   (δ = 0.895) but marginal tests may lack power.

5. **Development contamination risk.** The calibration seeds (29, 31) were used
   to set the rho = 0.95 dial; confirmatory seeds (41, 43) were reserved and
   never inspected during development.  The protocol is documented in
   `configs/preregistration.md`.

---

## Source repository

```
https://github.com/ayazkhan27/sma-1
```

Generating script: `scripts/confirmatory_battery.py --task ssb`

Confirmatory outputs: `reports/confirmatory/ssb_{rows,stats,summary}.csv`

Gate test: `tests/test_macfac.py::test_macfac_lattice_bridges_disjoint_vocabularies`

---

## Citation

```bibtex
@software{khan2026sma,
  author  = {Ayaz Khan},
  title   = {SMA-1: Structure-Mapping Agentic Memory},
  year    = {2026},
  license = {Apache-2.0},
  url     = {https://github.com/ayazkhan27/sma-1}
}
```

---

## License

Apache-2.0.  See `LICENSE` at the repository root.

Benchmark construction uses no third-party data; all functor names are randomly
generated.  The predicate lattice is fully synthetic.
