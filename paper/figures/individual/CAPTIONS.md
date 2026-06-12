# Individual figure captions (SMA-1 manuscript)

All figures are standalone (PDF+PNG). Schematics (1–4) are TikZ/xelatex
(scripts/figures_schematic_tikz.py); data figures (5–13) are Matplotlib +
SciencePlots from confirmatory CSVs (scripts/figures_supp.py). Statistics:
paired bootstrap (10k), Holm–Bonferroni, Cliff's δ; error bars are s.d. across
the five registered seeds unless noted.

1. **Architecture.** Deterministic write path (artifacts → versioned encoders →
   content-addressed case store → index) and certified read path (MAC bound →
   FAC structure mapping → receipts), with a single LLM gated outside memory
   under a cite-or-abstain policy.
2. **Representation.** A raw log session is encoded into a typed predicate DAG
   (entities, first- and higher-order relations); shared sub-expressions are
   hash-consed and the case is content-addressed (BLAKE3).
3. **Matcher.** The SME core: seed match hypotheses → close support into kernels
   → resolve kernel conflicts as a maximum-weight independent set (CP-SAT) →
   trickle-down systematicity scoring → project a provenance-tagged candidate
   inference.
4. **Cross-system mechanism.** A single mapping between a stored BGL incident and
   a lexically disjoint Spirit query: correspondences (dotted) form an injective
   gmap; non-identical functors match by bounded ascension through the predicate
   lattice (violet); the unmatched consequence is projected as a hypothetical
   candidate inference (red) — an object RAG cannot emit.
5. **Cross-system transfer (T1).** label-hit@1 across three legs × seven methods.
   SMA wins decisively on BGL→Thunderbird, leads with parity-to-hybrids on
   BGL→Spirit, null for all on HDFS→OpenStack.
6. **SSB structure-vs-surface (H2).** Under ground truth with vocabulary
   orthogonalized from structure, SMA forced-choice r1 = 1.000 and library
   r1 = 0.895 vs BM25/TF-IDF-dense r1 = 0.000 (δ = 0.895).
7. **Rare-event leverage (H4).** HDFS family-hit@5: on rare failure families SMA
   reaches 0.703 vs BM25 0.148 / dense 0.044 (δ up to 0.83) — ~5×.
8. **Within-system parity (H2).** BGL triage label-hit@1: SMA tied with dense and
   Hybrid-RRF (CIs include 0); no within-domain tax.
9. **Cross-domain reach (H5).** BugsInPy LOPO category accuracy: SMA 0.304/0.313
   vs BM25 0.186/0.216 and dense 0.113/0.109, far above random (0.094).
10. **Certified MAC/FAC (C1).** The content-vector bound U is admissible (U ≥ raw
    SES for every candidate pair; 0/1800 violations), licensing certified
    early-stopping; slack is non-negative throughout (conservative in magnitude).
11. **Calibration freeze.** Validation-only grid (scorer × normalization at
    ρ=0.95) over three metrics; the frozen choice (surprisal × max) is ringed.
12. **Liberty haystack.** needle-hit@5: SMA = dense = Hybrid-RRF = 0.995 (honest
    parity; out-of-corpus needles are lexically findable).
13. **Summary radar.** SMA vs the best baseline per task across all confirmatory
    tasks: SMA envelops the baseline on rare families, SSB, common families,
    Thunderbird and code; parity on triage and haystack.
