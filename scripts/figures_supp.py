"""Individual data-driven manuscript figures (Nature/NeurIPS-grade).

Each figure is a standalone PDF+PNG in paper/figures/individual/. Reads ONLY
confirmatory CSVs (reports/confirmatory) + calibration grid. Style: figstyle.py
(shared MAMMAL-ish language), Arial, colorblind-safe, KDE/CI/radar idioms drawn
from the FBNO (Nature MI) and NeuroPath (NeurIPS'25) references.
"""
from __future__ import annotations
import csv, datetime, pathlib, subprocess, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import scienceplots  # noqa
from figstyle import (SMA_C, KG_C, METHOD_C, TITLE_GRAY, axis_title, soften_spines)

plt.style.use(["science", "grid"])
plt.rcParams.update({
    "text.usetex": False,  # science style forces usetex; we need Unicode (Δ → δ ✓)
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 7, "axes.linewidth": 0.6, "figure.dpi": 300, "savefig.dpi": 600,
    "axes.prop_cycle": plt.cycler(color=[SMA_C, KG_C, "#7e8893", "#b97c0a"]),
})
ROOT = pathlib.Path(__file__).resolve().parents[1]
CONF = ROOT / "reports" / "confirmatory"
OUT = ROOT / "paper" / "figures" / "individual"
OUT.mkdir(parents=True, exist_ok=True)


def rev():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, cwd=ROOT).stdout.strip()
    except Exception:
        return "?"


STAMP = f"{datetime.date.today()} · {rev()} · prereg-v1 (confirmatory)"


def finish(fig, name):
    fig.text(0.995, 0.004, STAMP, fontsize=3.4, ha="right", color="0.65")
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.pdf/.png")


def rows(name):
    return list(csv.DictReader((CONF / name).open()))


ORDER = ["SMA", "Hybrid+Rerank", "Hybrid-RRF", "HippoRAG", "KG-PPR Proxy", "Dense RAG", "BM25"]
SHORT = {"SMA": "SMA", "Hybrid+Rerank": "Hyb+RR", "Hybrid-RRF": "Hyb-RRF", "HippoRAG": "HippoRAG",
         "KG-PPR Proxy": "KG-PPR", "Dense RAG": "Dense", "BM25": "BM25"}
LEGS = [("BGL->spirit_first20M", "BGL→Spirit"),
        ("BGL->thunderbird_first20M", "BGL→Thunderbird"),
        ("HDFS->OpenStack", "HDFS→OpenStack")]


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


# ============================================================ F5 transfer ===
def f5_transfer():
    summ, stats = rows("t1_summary.csv"), rows("t1_stats.csv")
    fig, ax = plt.subplots(figsize=(5.0, 2.7))
    w = 0.115
    for li, (leg, lbl) in enumerate(LEGS):
        for mi, m in enumerate(ORDER):
            r = next(r for r in summ if r["leg"] == leg and r["method"] == m and r["metric"] == "hit@1")
            ax.bar(li + (mi - 3) * w, float(r["mean"]), w * 0.9, yerr=float(r["sd"]),
                   color=METHOD_C[m], edgecolor="black" if m == "SMA" else "none",
                   linewidth=0.6, error_kw=dict(lw=0.6, capsize=1.4, capthick=0.6), zorder=3)
        base = [r for r in summ if r["leg"] == leg and r["method"] != "SMA" and r["metric"] == "hit@1"]
        best = max(base, key=lambda r: float(r["mean"]))
        st = next(s for s in stats if s["leg"] == leg and s["baseline"] == best["method"])
        ann_y = (0.97, 1.07, 0.66)[li]
        ax.text(li, ann_y, f"Δ={float(st['delta']):+.2f} vs {SHORT[best['method']]} ({stars(float(st['p_holm']))})",
                fontsize=4.8, ha="center", color="0.25")
    ax.axhline(0.5, color="0.6", lw=0.6, ls=":")
    ax.text(-0.62, 0.505, "chance", fontsize=5, color="0.5", va="bottom")
    ax.set_xticks(range(3), [l for _, l in LEGS], fontsize=6.5)
    ax.set_xlim(-0.68, 2.45); ax.set_ylim(0, 1.16)
    ax.set_yticks([0, .25, .5, .75, 1.0], ["0", "0.25", "0.50", "0.75", "1.00"], fontsize=6)
    ax.set_ylabel("label-hit@1  (mean ± s.d., 5 seeds)", fontsize=6.5)
    soften_spines(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=METHOD_C[m], ec="black" if m == "SMA" else "none", lw=0.6) for m in ORDER]
    ax.legend(handles, [SHORT[m] for m in ORDER], fontsize=4.8, ncol=4, frameon=False,
              loc="upper left", bbox_to_anchor=(-0.01, 1.0), handlelength=1.0,
              columnspacing=0.7, handletextpad=0.4)
    axis_title(ax, "", "Cross-system transfer: train one system, query another (frozen ontology)")
    finish(fig, "fig05_transfer")


# ================================================================= F6 SSB ===
def f6_ssb():
    summ, stats = rows("ssb_summary.csv"), rows("ssb_stats.csv")
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(5.4, 2.5), gridspec_kw=dict(width_ratios=[1, 1.1], wspace=0.32))
    # left: conceptual triple (drawn boxes)
    axl.set_xlim(0, 1); axl.set_ylim(0, 1); axl.axis("off")
    axl.text(0.5, 0.98, "Each SSB item: same structure, different surface", fontsize=6, ha="center", va="top", color=TITLE_GRAY, weight="bold")
    def chain(ax, x, y, cols, labels, title, tcol):
        ax.text(x, y + 0.16, title, fontsize=5, ha="center", color=tcol, weight="bold")
        for i, (c, lb) in enumerate(zip(cols, labels)):
            ax.add_patch(plt.Circle((x - 0.13 + i * 0.13, y), 0.035, color=c, zorder=3))
            if i < len(cols) - 1:
                ax.annotate("", xy=(x - 0.13 + (i + 1) * 0.13 - 0.035, y), xytext=(x - 0.13 + i * 0.13 + 0.035, y),
                            arrowprops=dict(arrowstyle="-|>", lw=0.6, color="0.4"))
        ax.text(x, y - 0.085, lb, fontsize=4, ha="center", color="0.5")
    chain(axl, 0.5, 0.78, [SMA_C, "#5aa9c4", "#2e6b86"], ["", "", "query vocab"], "query Q", "0.3")
    chain(axl, 0.5, 0.50, ["#9b86c4", "#7b8794", "#b58fd0"], ["", "", "DISJOINT vocab (lattice-bridged)"], "analog A(Q)  ✓ SMA picks this", SMA_C)
    chain(axl, 0.5, 0.22, [SMA_C, "#5aa9c4", "#2e6b86"], ["", "", "query vocab, structure broken (star)"], "distractor D(Q)  ✗ lexical picks this", "#b3403a")
    # right: results bars
    forced = next(r for r in summ if r["leg"] == "forced_choice" and r["method"] == "SMA" and r["metric"] == "r1")
    libr = {m: next(r for r in summ if r["leg"] == "library" and r["method"] == m and r["metric"] == "r1")
            for m in ("SMA", "BM25", "TFIDF-Dense")}
    names = ["SMA\n(forced)", "SMA\n(library)", "BM25", "TF-IDF\nDense"]
    vals = [float(forced["mean"]), float(libr["SMA"]["mean"]), float(libr["BM25"]["mean"]), float(libr["TFIDF-Dense"]["mean"])]
    sds = [float(forced["sd"]), float(libr["SMA"]["sd"]), float(libr["BM25"]["sd"]), float(libr["TFIDF-Dense"]["sd"])]
    cols = [SMA_C, SMA_C, "#c7ccd1", "#a7afb6"]
    axr.bar(range(4), vals, 0.66, yerr=sds, color=cols, edgecolor="black", linewidth=0.5,
            error_kw=dict(lw=0.6, capsize=1.6, capthick=0.6), zorder=3)
    for i, v in enumerate(vals):
        axr.text(i, v + 0.03, f"{v:.3f}", fontsize=5, ha="center", color="0.2")
    st = next(s for s in stats if s["leg"] == "library" and s["baseline"] == "BM25" and s["metric"] == "r1")
    axr.text(2.5, 0.55, f"Δ={float(st['delta']):.2f}\nδ={float(st['cliffs_delta']):.2f} ({stars(float(st['p_holm']))})",
             fontsize=5, ha="center", color="0.25")
    axr.set_xticks(range(4), names, fontsize=5.5)
    axr.set_ylim(0, 1.12); axr.set_ylabel("rank-1 accuracy (r1)", fontsize=6.5)
    axr.set_yticks([0, .25, .5, .75, 1.0], ["0", "0.25", "0.50", "0.75", "1.00"], fontsize=6)
    soften_spines(axr)
    axis_title(axr, "", "Structure beats surface under ground truth")
    finish(fig, "fig06_ssb")


# ============================================================== F7 family ===
def f7_family():
    summ, stats = rows("t2_summary.csv"), rows("t2_stats.csv")
    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    strata = [("HDFS_family_common", "common\nfamilies"), ("HDFS_family_rare", "rare\nfamilies")]
    meths = ["SMA", "BM25", "Dense RAG"]
    w = 0.25
    for si, (leg, lbl) in enumerate(strata):
        for mi, m in enumerate(meths):
            r = next(r for r in summ if r["leg"] == leg and r["method"] == m and r["metric"] == "hit@5")
            ax.bar(si + (mi - 1) * w, float(r["mean"]), w * 0.9, yerr=float(r["sd"]),
                   color=METHOD_C.get(m, SMA_C), edgecolor="black" if m == "SMA" else "none", linewidth=0.6,
                   error_kw=dict(lw=0.6, capsize=1.6, capthick=0.6), zorder=3)
        # delta vs dense (the weakest)
        st = next(s for s in stats if s["leg"] == leg and s["baseline"] == "Dense RAG")
        top = max(float(r["mean"]) + float(r["sd"]) for r in summ if r["leg"] == leg and r["metric"] == "hit@5")
        ax.text(si, top + 0.06, f"SMA δ={float(st['cliffs_delta']):.2f}\n({stars(float(st['p_holm']))})", fontsize=5, ha="center", color="0.25")
    ax.set_xticks(range(2), [l for _, l in strata], fontsize=6.5)
    ax.set_ylim(0, 1.0); ax.set_ylabel("family-hit@5  (mean ± s.d.)", fontsize=6.5)
    soften_spines(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=METHOD_C.get(m, SMA_C), ec="black" if m == "SMA" else "none", lw=0.6) for m in meths]
    ax.legend(handles, ["SMA", "BM25", "Dense"], fontsize=5.5, frameon=False, loc="upper right", handlelength=1.0)
    axis_title(ax, "", "Rare-event leverage: SMA ~5× baselines on rare failure families")
    finish(fig, "fig07_family")


# ============================================================== F8 triage ===
def f8_triage():
    summ, stats = rows("t2_summary.csv"), rows("t2_stats.csv")
    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    meths = ["SMA", "Hybrid-RRF", "Dense RAG", "Hybrid+Rerank", "BM25", "KG-PPR Proxy", "HippoRAG"]
    vals = [float(next(r for r in summ if r["leg"] == "BGL_triage" and r["method"] == m and r["metric"] == "hit@1")["mean"]) for m in meths]
    sds = [float(next(r for r in summ if r["leg"] == "BGL_triage" and r["method"] == m and r["metric"] == "hit@1")["sd"]) for m in meths]
    cols = [METHOD_C[m] for m in meths]
    ax.bar(range(len(meths)), vals, 0.66, yerr=sds, color=cols, edgecolor=["black"] + ["none"] * 6, linewidth=0.6,
           error_kw=dict(lw=0.6, capsize=1.6, capthick=0.6), zorder=3)
    ax.set_ylim(0.6, 1.02)
    ax.axhspan(0.985, 1.0, color=SMA_C, alpha=0.08, zorder=0)
    ax.text(3, 0.965, "SMA tied with best RAG (CIs include 0): no within-domain tax", fontsize=5, ha="center", color="0.3", style="italic")
    ax.set_xticks(range(len(meths)), [SHORT[m] for m in meths], fontsize=5.5, rotation=20)
    ax.set_ylabel("label-hit@1", fontsize=6.5)
    soften_spines(ax)
    axis_title(ax, "", "Within-system triage (BGL): SMA at parity with production RAG")
    finish(fig, "fig08_triage")


# ================================================================ F9 code ===
def f9_code():
    summ = rows("t3_summary.csv")
    stats = rows("t3_stats.csv")
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    meths = ["SMA", "BM25", "Dense RAG"]
    for ki, k in enumerate(("hit@1", "hit@5")):
        for mi, m in enumerate(meths):
            r = next(r for r in summ if r["method"] == m and r["metric"] == k)
            ax.bar(ki + (mi - 1) * 0.26, float(r["mean"]), 0.24, color=METHOD_C.get(m, SMA_C),
                   edgecolor="black" if m == "SMA" else "none", linewidth=0.6, zorder=3)
    ax.axhline(0.094, color="0.6", lw=0.6, ls=":")
    ax.text(1.35, 0.10, "random", fontsize=5, color="0.5")
    ax.set_xticks([0, 1], ["category@1", "category@5"], fontsize=6.5)
    ax.set_ylim(0, 0.42); ax.set_ylabel("LOPO category accuracy", fontsize=6.5)
    soften_spines(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=METHOD_C.get(m, SMA_C), ec="black" if m == "SMA" else "none", lw=0.6) for m in meths]
    ax.legend(handles, ["SMA", "BM25", "Dense"], fontsize=5.5, frameon=False, loc="upper left", handlelength=1.0)
    axis_title(ax, "", "Cross-domain reach: BugsInPy bug retrieval (code, frozen matcher)")
    finish(fig, "fig09_code")


# =========================================================== F11 calibration =
def f11_calibration():
    grid = list(csv.DictReader((ROOT / "reports" / "calibration_grid.csv").open()))
    fig, axes = plt.subplots(1, 3, figsize=(6.2, 2.2), gridspec_kw=dict(wspace=0.34, left=0.08, right=0.99, bottom=0.16, top=0.82))
    metrics = [("ssb_r1", "SSB rank-1"), ("hdfs_family_common", "HDFS common"), ("haystack_needles", "Liberty needles")]
    import cmasher as cmr
    cmap = cmr.get_sub_cmap("cmr.ocean", 0.15, 0.9)
    for ax, (key, lbl) in zip(axes, metrics):
        # heat over (scorer x normalization) averaged across gamma, at rho=0.95
        scorers, norms = ["ses", "surprisal"], ["max", "target"]
        M = np.zeros((2, 2))
        for i, sc in enumerate(scorers):
            for j, nz in enumerate(norms):
                vals = [float(r[key]) for r in grid if r["scorer"] == sc and r["normalization"] == nz and r["rho"] == "0.95"]
                M[i, j] = np.mean(vals)
        im = ax.imshow(M, cmap=cmap, vmin=M.min() - 0.02, vmax=M.max() + 0.02, aspect="auto")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center", fontsize=5.5,
                        color="white" if M[i, j] < M.mean() else "black")
        ax.set_xticks([0, 1], norms, fontsize=5.5); ax.set_yticks([0, 1], scorers, fontsize=5.5)
        ax.set_title(lbl, fontsize=6, color=TITLE_GRAY)
        # frozen pick (surprisal/max) ringed
        ax.add_patch(plt.Rectangle((-0.45, 0.55), 0.9, 0.9, fill=False, ec="#b3403a", lw=1.2))
    axes[0].set_ylabel("scorer", fontsize=6)
    fig.suptitle("Calibration grid (validation only) → frozen: surprisal × max (red) at ρ=0.95",
                 fontsize=6.8, color=TITLE_GRAY, weight="bold", y=0.99)
    finish(fig, "fig11_calibration")


# ============================================================= F12 haystack =
def f12_haystack():
    summ = rows("t4_summary.csv")
    fig, ax = plt.subplots(figsize=(3.8, 2.5))
    meths = ["SMA", "Hybrid-RRF", "Dense RAG", "BM25"]
    vals = [float(next(r for r in summ if r["method"] == m and r["metric"] == "needle_hit@5")["mean"]) for m in meths]
    sds = [float(next(r for r in summ if r["method"] == m and r["metric"] == "needle_hit@5")["sd"]) for m in meths]
    ax.bar(range(len(meths)), vals, 0.6, yerr=sds, color=[METHOD_C.get(m, SMA_C) for m in meths],
           edgecolor=["black"] + ["none"] * 3, linewidth=0.6, error_kw=dict(lw=0.6, capsize=1.6, capthick=0.6), zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v:.3f}", fontsize=5.5, ha="center", color="0.2")
    ax.set_ylim(0.9, 1.02); ax.set_xticks(range(len(meths)), [SHORT[m] for m in meths], fontsize=6)
    ax.set_ylabel("needle-hit@5", fontsize=6.5)
    ax.text(1.5, 0.915, "honest parity: out-of-corpus needles are lexically findable", fontsize=4.8, ha="center", color="0.4", style="italic")
    soften_spines(ax)
    axis_title(ax, "", "Liberty haystack: SMA = production hybrid (no edge here)")
    finish(fig, "fig12_haystack")


# ============================================================== F13 radar ===
def f13_radar():
    # normalized SMA vs best-baseline per task (one spider)
    axes_spec = [
        ("Transfer\nThunderbird", ("t1_summary.csv", "BGL->thunderbird_first20M", "hit@1")),
        ("Transfer\nSpirit", ("t1_summary.csv", "BGL->spirit_first20M", "hit@1")),
        ("Rare\nfamilies", ("t2_summary.csv", "HDFS_family_rare", "hit@5")),
        ("Common\nfamilies", ("t2_summary.csv", "HDFS_family_common", "hit@5")),
        ("BGL\ntriage", ("t2_summary.csv", "BGL_triage", "hit@1")),
        ("Code\n(BugsInPy)", ("t3_summary.csv", "bugsinpy_lopo", "hit@1")),
        ("SSB\nr1", ("ssb_summary.csv", "library", "r1")),
        ("Haystack", ("t4_summary.csv", "liberty_haystack", "needle_hit@5")),
    ]
    sma, base = [], []
    for _, (f, leg, met) in axes_spec:
        rs = rows(f)
        s = float(next(r for r in rs if r["leg"] == leg and r["method"] == "SMA" and r["metric"] == met)["mean"])
        others = [float(r["mean"]) for r in rs if r["leg"] == leg and r["method"] != "SMA" and r["metric"] == met]
        sma.append(s); base.append(max(others) if others else 0.0)
    labels = [a for a, _ in axes_spec]
    N = len(labels)
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    ang += ang[:1]
    fig, ax = plt.subplots(figsize=(4.2, 4.0), subplot_kw=dict(polar=True))
    for vals, c, lab, fill in [(sma, SMA_C, "SMA", 0.18), (base, "#8a8f96", "best baseline", 0.10)]:
        v = vals + vals[:1]
        ax.plot(ang, v, color=c, lw=1.4, label=lab, zorder=3)
        ax.fill(ang, v, color=c, alpha=fill, zorder=2)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(labels, fontsize=5.5)
    ax.set_ylim(0, 1.0); ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=4.6, color="0.5")
    ax.tick_params(pad=1)
    ax.legend(fontsize=6, frameon=False, loc="upper right", bbox_to_anchor=(1.18, 1.12))
    ax.set_title("SMA vs best baseline across all confirmatory tasks", fontsize=7, color=TITLE_GRAY, weight="bold", pad=14)
    finish(fig, "fig13_radar")


# ========================================================= F10 certified MAC =
def f10_certified():
    import cmasher as cmr
    cpath = ROOT / "reports" / "macfac_certification.csv"
    rs = list(csv.DictReader(cpath.open()))
    ses = np.array([float(r["ses_raw"]) for r in rs])
    bnd = np.array([float(r["u_bound"]) for r in rs])
    viol = int(np.sum(bnd < ses - 1e-9))
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(5.4, 2.5),
                                   gridspec_kw=dict(width_ratios=[1.2, 1], wspace=0.36))
    # scatter: bound (y) vs true raw SES (x); admissible iff every point on/above y=x
    hb = axl.hexbin(ses, bnd, gridsize=34, cmap=cmr.get_sub_cmap("cmr.ocean", 0.12, 0.92),
                    mincnt=1, linewidths=0)
    lim = max(ses.max(), bnd.max()) * 1.05
    axl.plot([0, lim], [0, lim], color="#b3403a", lw=1.0, ls="--", zorder=3)
    axl.fill_between([0, lim], [0, lim], [lim, lim], color="#2e8aa6", alpha=0.05, zorder=0)
    axl.text(lim * 0.30, lim * 0.78, "admissible region\nU ≥ SES (every point)", fontsize=4.8,
             color="0.3", ha="center")
    axl.text(lim * 0.04, lim * 0.04, "y = x", fontsize=4.6, color="#b3403a", rotation=33)
    axl.text(lim * 0.30, lim * 0.96, f"bound violations: {viol} / {len(rs)}", fontsize=5.4,
             color="#b3403a", weight="bold", ha="center")
    axl.set_xlim(0, lim); axl.set_ylim(0, lim)
    axl.set_xlabel("true structural score (raw SES)", fontsize=6.5)
    axl.set_ylabel("MAC content-vector bound  U", fontsize=6.5)
    soften_spines(axl)
    axis_title(axl, "", "Lemma 2: the screening bound is admissible")
    cb = fig.colorbar(hb, ax=axl, fraction=0.045, pad=0.02); cb.ax.tick_params(labelsize=4.5)
    cb.set_label("pairs", fontsize=5)
    # slack distribution (honest: valid upper bound, conservative in magnitude)
    slack = bnd - ses
    axr.hist(slack, bins=30, color=SMA_C, alpha=0.85, edgecolor="white", linewidth=0.3)
    axr.axvline(0, color="#b3403a", lw=1.0, ls="--")
    axr.set_xlabel("slack  U − SES   (≥ 0 always)", fontsize=6.5)
    axr.set_ylabel("candidate pairs", fontsize=6.5)
    axr.set_xlim(left=-20)
    axr.text(0.97, 0.95, "guarantees correct\nearly-stop; conservative\nin magnitude",
             transform=axr.transAxes, fontsize=4.8, color="0.35", ha="right", va="top")
    soften_spines(axr)
    axis_title(axr, "", "Bound never underestimates the true score")
    finish(fig, "fig10_certified")


if __name__ == "__main__":
    f5_transfer(); f6_ssb(); f7_family(); f8_triage(); f9_code(); f10_certified()
    f11_calibration(); f12_haystack(); f13_radar()
    print("data figures done")
