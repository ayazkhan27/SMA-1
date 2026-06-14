"""Publication data figures (matplotlib + SciencePlots 'nature'), exported as SVG.

Produces the figures whose role is mine (quantitative), from committed CSVs:
  Figure 2     main agentic results (SMA vs enterprise RAG/KG, per domain)
  Figure 4d    boundary data (de-risk: tie oracle / beat RAG; flat-tabular null)
  ED1          SSB + cross-system transfer (the engine, in isolation)
  ED3          calibration / ablation grid (frozen-dial selection)

Outputs -> paper/figures/svg/<name>.svg (+ .png for QA).
  python3 scripts/figures_paper.py
"""
from __future__ import annotations

import csv
import pathlib
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  (registers the 'nature' style)

plt.style.use(["nature", "no-latex"])
# Publication readability: SciencePlots 'nature' defaults to 5-7 pt, which is too
# small at \linewidth. Bump the whole type ramp so every label is legible in print
# without overlapping the diagram (cowplot/patchwork-style legible panels).
plt.rcParams.update({
    "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 8.0,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
})

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONF = ROOT / "reports/confirmatory"
OUT = ROOT / "paper/figures/svg"
OUT.mkdir(parents=True, exist_ok=True)

SMA_C = "#2E8AA6"
GRAYS = {"bm25": "#C7CCD1", "dense": "#A7AFB6", "hybrid_rrf": "#7E8893",
         "hybrid_rerank": "#5B6670"}
GOLD = "#D39A3E"
MEM_LABEL = {"sma": "SMA", "bm25": "BM25", "dense": "Dense-RAG",
             "hybrid_rrf": "Hybrid-RRF", "hybrid_rerank": "Hybrid+Rerank", "hipporag": "HippoRAG"}
MEM_COLOR = {"sma": SMA_C, **GRAYS, "hipporag": GOLD}
MEM_ORDER = ["sma", "bm25", "dense", "hybrid_rrf", "hybrid_rerank", "hipporag"]
ARM_ORDER = [("medicine", "Medicine"), ("discovery", "Genomics"),
             ("finance", "Finance"), ("cyber", "Cyber"), ("legal", "Legal")]


def _rows(path):
    return list(csv.DictReader(open(path))) if path.exists() else []


def _save(fig, name):
    for ext in ("svg", "png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    return OUT / f"{name}.svg"


def _panel(ax, letter, *, x=-0.13, y=1.08):
    """Cowplot/patchwork-style bold panel tag in the axes' top-left corner."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left", color="#1A1F24")


# ---------------------------------------------------------------- Figure 2 ---
def figure2():
    arms = {a: {r["memory"]: r for r in _rows(CONF / f"agentic_{a}.csv")}
            for a, _ in ARM_ORDER}
    present = [(a, lab) for a, lab in ARM_ORDER if arms.get(a)]
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.4), gridspec_kw={"width_ratios": [2.2, 1]})

    # (a) grouped bars: tail top-5 (rare slice; all-slice where the rare slice is
    # undefined — CPC/legal has near-uniform IC so no rare split, flagged with †).
    def t5(a, m):
        r = arms[a].get(m)
        if not r:
            return 0.0
        use_all = int(arms[a]["sma"]["n_rare"]) == 0
        return float(r["t5_all" if use_all else "t5_rare"])
    daggers = {a for a, _ in present if int(arms[a]["sma"]["n_rare"]) == 0}
    mems = MEM_ORDER
    nb = len(mems); w = 0.82 / nb
    for i, m in enumerate(mems):
        vals = [t5(a, m) for a, _ in present]
        xs = [j + (i - nb / 2) * w + w / 2 for j in range(len(present))]
        ax[0].bar(xs, vals, w, color=MEM_COLOR[m], label=MEM_LABEL[m],
                  edgecolor="white", linewidth=0.3)
    ax[0].set_xticks(range(len(present)))
    ax[0].set_xticklabels([l + ("†" if a in daggers else "") for a, l in present])
    ax[0].set_ylabel("Tail top-5 accuracy (rare slice)"); ax[0].set_ylim(0, 1.0)
    ax[0].legend(fontsize=7.5, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.13),
                 frameon=False, columnspacing=1.0, handlelength=1.2, handletextpad=0.4)
    if daggers:
        ax[0].text(0.0, -0.30, "† all-query slice (rare slice undefined for CPC's uniform IC)",
                   transform=ax[0].transAxes, fontsize=6.5, color="#5F6B78", style="italic")
    ax[0].set_title("SMA vs RAG/KG baselines, by domain", loc="left", fontsize=9.5, pad=8)
    _panel(ax[0], "a")

    # (b) effect-size forest: delta(SMA - best RAG) per domain with 95% CI
    ys = range(len(present))
    d = [float(arms[a]["sma"]["primary_delta_t5"]) for a, _ in present]
    lo = [d[i] - float(arms[a]["sma"]["primary_ci_low"]) for i, (a, _) in enumerate(present)]
    hi = [float(arms[a]["sma"]["primary_ci_high"]) - d[i] for i, (a, _) in enumerate(present)]
    ax[1].errorbar(d, list(ys), xerr=[lo, hi], fmt="o", color=SMA_C, capsize=2,
                   markersize=4, linewidth=1.0)
    ax[1].axvline(0, color="#9AA3AB", lw=0.6, ls="--")
    ax[1].set_yticks(list(ys)); ax[1].set_yticklabels([l for _, l in present])
    ax[1].set_xlabel(r"$\Delta$ tail top-5 (SMA $-$ best RAG)")
    ax[1].set_title("Effect size, 95% CI", loc="left", fontsize=9.5, pad=8)
    _panel(ax[1], "b", x=-0.30)
    fig.tight_layout(w_pad=1.6)
    fig.subplots_adjust(bottom=0.24)
    return _save(fig, "figure2_results")


# ---------------------------------------------------------------- Figure 4d --
def figure4_data():
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.3))
    # (a) de-risk: rare-slice top-5, SMA vs oracle vs RAG/KG, on HPO + GO
    suite = [r for r in _rows(CONF / "ontology_suite.csv") if r["slice"] == "rare"]
    labels = {"A1-hpo-rare-disease": "HPO\n(rare disease)", "A2-go-gene-function": "GO\n(gene function)"}
    methods = [("sma_t5", "SMA", SMA_C), ("phen_t5", "Phenomizer\n(ontology oracle)", "#9B86C4"),
               ("dense_t5", "Dense-RAG", "#A7AFB6"), ("hippo_t5", "HippoRAG", GOLD),
               ("jac_t5", "Jaccard", "#C7CCD1")]
    arms = [r for r in suite]
    nb = len(methods); w = 0.82 / nb
    for i, (col, lab, c) in enumerate(methods):
        vals = [float(r[col]) for r in arms]
        xs = [j + (i - nb / 2) * w + w / 2 for j in range(len(arms))]
        ax[0].bar(xs, vals, w, color=c, label=lab, edgecolor="white", linewidth=0.3)
    ax[0].set_xticks(range(len(arms))); ax[0].set_xticklabels([labels[r["arm"]] for r in arms])
    ax[0].set_ylabel("Top-5 accuracy (rare slice)"); ax[0].set_ylim(0, 1.0)
    ax[0].legend(fontsize=6.8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.14),
                 frameon=False, columnspacing=1.0, handlelength=1.2)
    ax[0].set_title("Ties the oracle, beats RAG/KG", loc="left", fontsize=9.5, pad=8)
    _panel(ax[0], "a")

    # (b) flat-tabular null: SMA F1 vs baselines on diabetes / fraud (4b)
    pts = []
    for dom, f in (("Diabetes", "cd_diabetes_before.csv"), ("Card fraud", "cd_ieee_before.csv")):
        rows = _rows(CONF / f)
        if rows:
            sma = float(rows[0]["sma_f1"])
            base = statistics.mean(float(r["baseline_f1"]) for r in rows)
            pts.append((dom, sma, base))
    x = range(len(pts)); w2 = 0.36
    ax[1].bar([i - w2 / 2 for i in x], [p[1] for p in pts], w2, color=SMA_C, label="SMA", edgecolor="white", lw=0.3)
    ax[1].bar([i + w2 / 2 for i in x], [p[2] for p in pts], w2, color="#A7AFB6", label="value-based RAG", edgecolor="white", lw=0.3)
    ax[1].set_xticks(list(x)); ax[1].set_xticklabels([p[0] for p in pts])
    ax[1].set_ylabel("macro-F1"); ax[1].set_ylim(0, 0.8)
    ax[1].legend(fontsize=7.5, frameon=False, loc="upper right")
    ax[1].set_title("Flat-tabular null (no ontology)", loc="left", fontsize=9.5, pad=8)
    _panel(ax[1], "b")
    ax[1].text(0.5, 0.04, "no structure to exploit → SMA = baseline", transform=ax[1].transAxes,
               fontsize=7, ha="center", style="italic", color="#5F6B78")
    fig.tight_layout(w_pad=1.8)
    fig.subplots_adjust(bottom=0.22)
    return _save(fig, "figure4_boundary_data")


# ---------------------------------------------------------------- ED1 --------
def ed1_ssb():
    """SSB: structure-matching across a zero-lexical-overlap vocabulary gap.
    Lexical/dense baselines genuinely score rank-1 = 0 (nothing to match on
    surface). The transfer panel is deferred: it needs a paired per-leg
    comparison, not a leg-averaged macro-F1 (which biases across legs)."""
    rows = [r for r in _rows(CONF / "ssb_summary.csv") if r["metric"] == "r1"]
    if not rows:
        return None
    legs = sorted(set(r["leg"] for r in rows))
    methods = sorted(set(r["method"] for r in rows), key=lambda m: (m != "SMA", m))
    fig, ax = plt.subplots(figsize=(4.7, 3.2))
    nb = len(methods); w = 0.78 / nb
    mc = {"SMA": SMA_C, "BM25": "#C7CCD1", "TFIDF-Dense": "#A7AFB6", "Dense": "#A7AFB6"}
    for i, m in enumerate(methods):
        vals = [next((float(r["mean"]) for r in rows if r["leg"] == lg and r["method"] == m), 0) for lg in legs]
        xs = [j + (i - nb / 2) * w + w / 2 for j in range(len(legs))]
        ax.bar(xs, vals, w, color=mc.get(m, "#888"), label=m, edgecolor="white", lw=0.3)
        # value labels make the deliberate zeros legible (the whole point)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5,
                    color=(SMA_C if m == "SMA" else "#8A929B"))
    ax.set_xticks(range(len(legs))); ax.set_xticklabels([l.replace("_", " ") for l in legs])
    ax.set_ylabel("rank-1 accuracy"); ax.set_ylim(0, 1.14)
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.13))
    ax.set_title("SSB: only structure survives a zero-lexical-overlap gap", loc="left", fontsize=9)
    ax.text(0.5, 0.42, "BM25 & dense retrieval score 0 —\nno surface overlap to match;\nonly structure-mapping solves it",
            transform=ax.transAxes, fontsize=7, style="italic", color="#5F6B78", ha="center")
    fig.tight_layout()
    return _save(fig, "ed1_ssb")


# ---------------------------------------------------------------- ED3 --------
def ed3_ablation():
    """Why the frozen dials: ablate one knob at a time from the chosen config and
    show it loses on a validation metric. Frozen = surprisal/max/gamma0.25/rho0.95."""
    rows = _rows(ROOT / "reports/calibration_grid.csv")
    if not rows:
        return None

    def cfg(scorer, norm, rho, gamma="0.25"):
        return next((r for r in rows if r["scorer"] == scorer and r["normalization"] == norm
                     and r["rho"] == rho and r["gamma"] == gamma), None)

    configs = [("Frozen (surprisal·max·ρ0.95)", cfg("surprisal", "max", "0.95"), SMA_C),
               ("−ρ → 0.90", cfg("surprisal", "max", "0.9"), "#5B6670"),
               ("−norm → target", cfg("surprisal", "target", "0.95"), "#7E8893"),
               ("−scorer → SES", cfg("ses", "max", "0.95"), "#C7CCD1")]
    configs = [(lab, r, c) for lab, r, c in configs if r]
    metrics = [("ssb_r1", "SSB\nrank-1"), ("hdfs_family_common", "Family-hit\n(common)"),
               ("haystack_needles", "Haystack\nrecall")]
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    nb = len(configs); w = 0.8 / nb
    for i, (lab, r, c) in enumerate(configs):
        vals = [float(r[m]) for m, _ in metrics]
        xs = [j + (i - nb / 2) * w + w / 2 for j in range(len(metrics))]
        ax.bar(xs, vals, w, color=c, label=lab, edgecolor="white", lw=0.3)
    ax.set_xticks(range(len(metrics))); ax.set_xticklabels([l for _, l in metrics])
    ax.set_ylabel("validation score"); ax.set_ylim(0, 1.1)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.42))
    ax.set_title("Frozen dials win the calibration grid", loc="left", fontsize=9.5)
    ax.annotate("ρ=0.90 breaks\nstructure-only SSB", xy=(-0.27, 0.79), xytext=(0.15, 0.45),
                fontsize=7, color="#5F6B78", ha="left",
                arrowprops=dict(arrowstyle="->", color="#9AA3AB", lw=0.6))
    fig.tight_layout()
    return _save(fig, "ed3_ablation")


# ---------------------------------------------------------------- Figure 5 ---
def figure5_trustworthy():
    """Phase 5 LLM-QA: the SMA-grounded agent is a *verifiable specialist*.

    (a) the trustworthy-QA axes across the three memory conditions
        (closed-book / Dense-RAG / SMA) — accuracy plus the capability axes RAG
        structurally lacks (citation-faithfulness, abstention, known-vs-unknown
        discrimination, novelty-F1); N/A cells (closed-book has no retrieval) are
        marked, not zeroed.
    (b) the mechanism: SMA's RAW structural grounding score cleanly separates
        known (answerable) from held-out (unknown) cases, so the calibrated gate
        abstains and flags novelty; the dashed line is the calibrated threshold.
    """
    def summ(mem):
        rows = _rows(CONF / f"qa_{mem}_summary.csv")
        return rows[0] if rows else None

    s = {m: summ(m) for m in ("none", "dense", "sma")}
    if not s["sma"]:
        return None

    def val(mem, key):
        r = s.get(mem)
        if not r:
            return None
        v = r.get(key, "NA")
        return None if v in ("NA", "", None) else float(v)

    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.5), gridspec_kw={"width_ratios": [1.55, 1]})

    # (a) grouped bars over the trustworthy axes
    axes = [("accuracy", "Accuracy"), ("citation_faithfulness", "Citation\nfaithful"),
            ("abstain_recall", "Abstain\nrecall"), ("grounding_auroc", "Known vs\nunknown"),
            ("novelty_f1", "Novelty\nF1")]
    conds = [("none", "Closed-book", "#C7CCD1"), ("dense", "Dense-RAG", "#A7AFB6"),
             ("sma", "SMA", SMA_C)]
    nb = len(conds); w = 0.8 / nb
    for i, (mem, lab, c) in enumerate(conds):
        labelled = False
        for j, (key, _) in enumerate(axes):
            v = val(mem, key)
            x = j + (i - nb / 2) * w + w / 2
            if v is None:
                ax[0].text(x, 0.02, "n/a", ha="center", va="bottom", fontsize=6.5,
                           rotation=90, color="#9AA3AB")
            else:
                ax[0].bar(x, v, w, color=c, edgecolor="white", linewidth=0.3,
                          label=(lab if not labelled else None))
                labelled = True
    ax[0].set_xticks(range(len(axes))); ax[0].set_xticklabels([l for _, l in axes])
    ax[0].set_ylabel("score"); ax[0].set_ylim(0, 1.05)
    ax[0].axhline(0.5, color="#E3E7EA", lw=0.5, zorder=0)
    ax[0].legend(fontsize=7.5, ncol=3, frameon=False, loc="upper center",
                 bbox_to_anchor=(0.5, -0.16), columnspacing=1.2, handlelength=1.2)
    ax[0].set_title("Trustworthy-QA across memories", loc="left", fontsize=9.5, pad=8)
    _panel(ax[0], "a")

    # (b) mechanism: SMA grounding-score separation, known vs unknown
    import numpy as np
    rows = _rows(CONF / "qa_sma.csv")
    ans = [float(r["grounding_score"]) for r in rows
           if str(r["answerable"]).strip().lower() == "true"
           and r["grounding_score"] not in ("", "NA", "None")]
    held = [float(r["grounding_score"]) for r in rows
            if str(r["answerable"]).strip().lower() == "false"
            and r["grounding_score"] not in ("", "NA", "None")]
    thr = val("sma", "score_threshold")
    auc = val("sma", "grounding_auroc")
    if ans and held:
        bins = np.linspace(min(ans + held), max(ans + held), 20)
        ax[1].hist(held, bins=bins, color="#A7AFB6", alpha=0.85,
                   label="held-out (unknown)", edgecolor="white", linewidth=0.2)
        ax[1].hist(ans, bins=bins, color=SMA_C, alpha=0.7,
                   label="answerable (known)", edgecolor="white", linewidth=0.2)
        ymax = ax[1].get_ylim()[1]
        ax[1].set_ylim(0, ymax * 1.28)  # headroom for the legend in the central gap
        if thr is not None:
            ax[1].axvline(thr, color="#D98A3D", lw=1.2, ls="--")
            ax[1].text(thr, ymax * 0.42, "calibrated\nthreshold ", rotation=90,
                       fontsize=7, color="#B86E1F", ha="right", va="center")
        if auc is not None:
            ax[1].text(0.5, 0.55, f"AUROC\n{auc:.2f}", transform=ax[1].transAxes,
                       ha="center", fontsize=9, color=SMA_C, fontweight="bold")
        ax[1].set_xlabel("SMA structural grounding score (top hit)")
        ax[1].set_ylabel("cases")
        # legend in the top-centre whitespace (the gap between the two modes)
        ax[1].legend(fontsize=7, frameon=False, loc="upper center", ncol=1,
                     handlelength=1.1, borderaxespad=0.2)
    ax[1].set_title("Structure separates known from unknown", loc="left", fontsize=9.5, pad=8)
    _panel(ax[1], "b", x=-0.22)
    fig.tight_layout(w_pad=1.8)
    fig.subplots_adjust(bottom=0.22)
    return _save(fig, "figure5_trustworthy")


if __name__ == "__main__":
    for fn in (figure2, figure4_data, ed1_ssb, ed3_ablation, figure5_trustworthy):
        try:
            p = fn()
            print(f"wrote {p}" if p else f"{fn.__name__}: no data, skipped")
        except Exception as e:
            print(f"{fn.__name__}: ERROR {type(e).__name__}: {e}")
