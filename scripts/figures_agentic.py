"""Agentic-suite results figure (Nature-MI grade, multi-panel).

Reads reports/confirmatory/agentic_<arm>.csv and renders a 4-panel figure:
  a  per-domain tail top-5 (rare slice): SMA vs the enterprise RAG/KG gauntlet
  b  cite-or-abstain: risk-coverage AURC per memory (lower = better calibrated)
  c  structural novelty F1 per memory (only structure-aware methods are non-zero)
  d  headline effect: delta(SMA - best enterprise RAG) tail top-5 with 95% CI

  python3 scripts/figures_agentic.py
"""
from __future__ import annotations

import csv
import pathlib
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.figstyle import SMA_C, TITLE_GRAY, axis_title, soften_spines

OUT = pathlib.Path("reports/confirmatory")
FIGDIR = pathlib.Path("paper/figures/individual")
ARMS = ["medicine", "cyber", "discovery", "legal", "finance"]
MEM_ORDER = ["sma", "bm25", "dense", "hybrid_rrf", "hybrid_rerank", "hipporag"]
MEM_LABEL = {"sma": "SMA", "bm25": "BM25", "dense": "Dense-RAG",
             "hybrid_rrf": "Hybrid-RRF", "hybrid_rerank": "Hybrid+Rerank", "hipporag": "HippoRAG"}
MEM_COLOR = {"sma": SMA_C, "bm25": "#c7ccd1", "dense": "#a7afb6", "hybrid_rrf": "#7e8893",
             "hybrid_rerank": "#5b6670", "hipporag": "#d39a3e"}


def load_arms():
    out = {}
    for arm in ARMS:
        p = OUT / f"agentic_{arm}.csv"
        if not p.exists():
            continue
        rows = {r["memory"]: r for r in csv.DictReader(p.open())}
        out[arm] = rows
    return out


def main():
    data = load_arms()
    if not data:
        print("no agentic_*.csv found"); return
    arms = [a for a in ARMS if a in data]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    aa, ab, ac, ad = axes.ravel()

    # --- a: per-domain tail top-5 (rare) grouped bars ---------------------
    mems = [m for m in MEM_ORDER if any(m in data[a] for a in arms)]
    n = len(mems); w = 0.8 / n
    for i, m in enumerate(mems):
        vals = [float(data[a][m]["t5_rare"]) if m in data[a] else 0 for a in arms]
        xs = [j + (i - n / 2) * w + w / 2 for j in range(len(arms))]
        aa.bar(xs, vals, w, color=MEM_COLOR[m], label=MEM_LABEL[m],
               edgecolor="white", linewidth=0.4)
    aa.set_xticks(range(len(arms))); aa.set_xticklabels([a.capitalize() for a in arms])
    aa.set_ylabel("Tail top-5 accuracy (rare slice)"); aa.set_ylim(0, 1.0)
    aa.legend(fontsize=5.2, ncol=3, loc="upper center", frameon=False,
              bbox_to_anchor=(0.5, -0.13))
    axis_title(aa, "a", "SMA vs enterprise RAG/KG, by domain"); soften_spines(aa)

    # --- b: cite-or-abstain AURC (lower better) ---------------------------
    arm0 = arms[0]
    bm = [m for m in MEM_ORDER if m in data[arm0]]
    aurc = [float(data[arm0][m]["aurc"]) for m in bm]
    ab.barh([MEM_LABEL[m] for m in bm], aurc, color=[MEM_COLOR[m] for m in bm],
            edgecolor="white", linewidth=0.4)
    ab.invert_yaxis(); ab.set_xlabel("Risk-coverage AURC  (lower = better)")
    axis_title(ab, "b", f"Cite-or-abstain calibration ({arm0})"); soften_spines(ab)

    # --- c: novelty F1 ----------------------------------------------------
    nf1 = [float(data[arm0][m]["novelty_f1"]) for m in bm]
    ac.barh([MEM_LABEL[m] for m in bm], nf1, color=[MEM_COLOR[m] for m in bm],
            edgecolor="white", linewidth=0.4)
    ac.invert_yaxis(); ac.set_xlabel("Novelty F1  (flagging the unknown)")
    ac.text(0.97, 0.04, "pure RAG = 0\n(no novelty signal)", transform=ac.transAxes,
            fontsize=5.4, color=TITLE_GRAY, ha="right", va="bottom", style="italic")
    axis_title(ac, "c", "Structural-novelty detection"); soften_spines(ac)

    # --- d: headline delta (SMA - best RAG) with 95% CI -------------------
    deltas, los, his, labs = [], [], [], []
    for a in arms:
        r = data[a]["sma"]
        deltas.append(float(r["primary_delta_t5"]))
        los.append(float(r["primary_delta_t5"]) - float(r["primary_ci_low"]))
        his.append(float(r["primary_ci_high"]) - float(r["primary_delta_t5"]))
        labs.append(f"{a.capitalize()}\nvs {r['best_enterprise']}")
    ys = range(len(arms))
    ad.errorbar(deltas, ys, xerr=[los, his], fmt="o", color=SMA_C, capsize=2.5,
                markersize=4, linewidth=1.0)
    ad.axvline(0, color="#9aa3ab", linewidth=0.6, linestyle="--")
    ad.set_yticks(list(ys)); ad.set_yticklabels(labs, fontsize=5.6)
    ad.set_xlabel(r"$\Delta$ tail top-5 (SMA $-$ best RAG), 95% CI")
    axis_title(ad, "d", "Headline effect size per domain"); soften_spines(ad)

    fig.tight_layout(w_pad=1.5, h_pad=2.0)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"fig_agentic_results.{ext}", bbox_inches="tight")
    print(f"wrote {FIGDIR}/fig_agentic_results.pdf (+png); arms: {arms}")


if __name__ == "__main__":
    main()
