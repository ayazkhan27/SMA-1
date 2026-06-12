# Figure 4 — Cross-system transfer (headline confirmatory result)

**Role:** the money figure — H1 on real data. Train on one system, query a
different one with disjoint vocabulary; SMA wins because failure *motifs* recur
even when words don't. MAMMAL Fig 1E is the analog (method-vs-SOTA comparison).

**Toolchain:** Matplotlib (SciencePlots), CMasher categorical for the 7 methods,
**matplotlib-label-lines** or a compact legend, plus a companion **autorank /
scikit-posthocs critical-difference (CD) diagram** as panel C.

## Panel A — grouped bars, the three transfer legs
x = three legs (BGL→Spirit, BGL→Thunderbird, HDFS→OpenStack). For each leg, 7
bars: SMA (deep teal, heaviest), BM25 / Dense / Hybrid-RRF / Hybrid+Rerank
(grayscale ramp), KG-PPR / HippoRAG (amber). Metric = macro-F1 (and a twin for
label-hit@1). **95% paired-bootstrap CIs** as caps; significance brackets
(Holm-corrected) between SMA and the best baseline per leg. The "collapse line"
at macro-F1 = 0.3333 (two-class collapse) drawn faint — baselines hugging it is
the story.

## Panel B — the gap, per-query
A paired dot/slope panel or violin: per-query SMA−baseline deltas pooled across
seeds 201–205, showing the distribution (not just the mean). Cliff's δ annotated.
brokenaxes if the SMA-vs-dense gap dwarfs the within-baseline spread.

## Panel C — critical-difference diagram
autorank CD diagram ranking all 7 methods across all legs×seeds: methods on a
rank axis, a crossbar joining those *not* significantly different. SMA sits
alone at the top → the cleanest possible "it wins" visual.

**Data sources:** `reports/confirmatory/t1_summary.csv`, `t1_rows.csv` (per-query),
`t1_stats.csv` (bootstrap/Holm/Cliff's δ). All hatched until the battery's T1 is
confirmed (already complete as of this writing — un-hatch on read).
**Caption point:** "Under a frozen ontology and zero target-system tuning, SMA
transfers across systems with disjoint vocabularies where lexical and dense
retrieval collapse toward two-class chance."
