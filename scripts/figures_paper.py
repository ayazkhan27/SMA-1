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
plt.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})

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
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    return OUT / f"{name}.svg"


# ---------------------------------------------------------------- Figure 2 ---
def figure2():
    arms = {a: {r["memory"]: r for r in _rows(CONF / f"agentic_{a}.csv")}
            for a, _ in ARM_ORDER}
    present = [(a, lab) for a, lab in ARM_ORDER if arms.get(a)]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [2.2, 1]})

    # (a) grouped bars: tail top-5 (rare) per domain
    mems = MEM_ORDER
    nb = len(mems); w = 0.82 / nb
    for i, m in enumerate(mems):
        vals = [float(arms[a][m]["t5_rare"]) if m in arms[a] else 0 for a, _ in present]
        xs = [j + (i - nb / 2) * w + w / 2 for j in range(len(present))]
        ax[0].bar(xs, vals, w, color=MEM_COLOR[m], label=MEM_LABEL[m],
                  edgecolor="white", linewidth=0.3)
    ax[0].set_xticks(range(len(present))); ax[0].set_xticklabels([l for _, l in present])
    ax[0].set_ylabel("Tail top-5 accuracy (rare slice)"); ax[0].set_ylim(0, 1.0)
    ax[0].legend(fontsize=5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    ax[0].set_title("a   SMA vs enterprise RAG/KG, by domain", loc="left", fontsize=8)

    # (b) effect-size forest: delta(SMA - best RAG) per domain with 95% CI
    ys = range(len(present))
    d = [float(arms[a]["sma"]["primary_delta_t5"]) for a, _ in present]
    lo = [d[i] - float(arms[a]["sma"]["primary_ci_low"]) for i, (a, _) in enumerate(present)]
    hi = [float(arms[a]["sma"]["primary_ci_high"]) - d[i] for i, (a, _) in enumerate(present)]
    ax[1].errorbar(d, list(ys), xerr=[lo, hi], fmt="o", color=SMA_C, capsize=2,
                   markersize=4, linewidth=1.0)
    ax[1].axvline(0, color="#9AA3AB", lw=0.6, ls="--")
    ax[1].set_yticks(list(ys)); ax[1].set_yticklabels([l for _, l in present], fontsize=6)
    ax[1].set_xlabel(r"$\Delta$ tail top-5 (SMA $-$ best RAG)")
    ax[1].set_title("b   Effect size, 95% CI", loc="left", fontsize=8)
    fig.tight_layout(w_pad=1.5)
    return _save(fig, "figure2_results")


# ---------------------------------------------------------------- Figure 4d --
def figure4_data():
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
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
    ax[0].set_xticks(range(len(arms))); ax[0].set_xticklabels([labels[r["arm"]] for r in arms], fontsize=6)
    ax[0].set_ylabel("Top-5 accuracy (rare slice)"); ax[0].set_ylim(0, 1.0)
    ax[0].legend(fontsize=4.6, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)
    ax[0].set_title("a   Ties the oracle, beats RAG/KG", loc="left", fontsize=8)

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
    ax[1].legend(fontsize=5.5, frameon=False, loc="upper right")
    ax[1].set_title("b   Flat-tabular null (no ontology)", loc="left", fontsize=8)
    ax[1].text(0.5, 0.04, "no structure to exploit → SMA = baseline", transform=ax[1].transAxes,
               fontsize=5.4, ha="center", style="italic", color="#5F6B78")
    fig.tight_layout(w_pad=1.8)
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
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    nb = len(methods); w = 0.8 / nb
    mc = {"SMA": SMA_C, "BM25": "#C7CCD1", "TFIDF-Dense": "#A7AFB6", "Dense": "#A7AFB6"}
    for i, m in enumerate(methods):
        vals = [next((float(r["mean"]) for r in rows if r["leg"] == lg and r["method"] == m), 0) for lg in legs]
        xs = [j + (i - nb / 2) * w + w / 2 for j in range(len(legs))]
        ax.bar(xs, vals, w, color=mc.get(m, "#888"), label=m, edgecolor="white", lw=0.3)
    ax.set_xticks(range(len(legs))); ax.set_xticklabels([l.replace("_", " ") for l in legs])
    ax.set_ylabel("rank-1 accuracy"); ax.set_ylim(0, 1.08)
    ax.legend(fontsize=5.5, frameon=False, loc="upper right")
    ax.set_title("SSB: structure across a zero-lexical-overlap gap", loc="left", fontsize=8)
    ax.text(0.02, 0.06, "lexical & dense baselines: rank-1 = 0\n(no surface overlap to exploit)",
            transform=ax.transAxes, fontsize=5.6, style="italic", color="#5F6B78")
    fig.tight_layout()
    return _save(fig, "ed1_ssb")


# ---------------------------------------------------------------- ED3 --------
def ed3_ablation():
    rows = _rows(ROOT / "reports/calibration_grid.csv")
    if not rows:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    # at normalization=max: family_rare vs rho, one line per scorer
    sub = [r for r in rows if r["normalization"] == "max"]
    SCOL = {"surprisal": SMA_C, "ses": "#A7AFB6", "mdl": GOLD}
    scorers = sorted(set(r["scorer"] for r in sub))
    for sc in scorers:
        srows = sorted((r for r in sub if r["scorer"] == sc), key=lambda r: float(r["rho"]))
        # average across gamma for each rho
        rhos = sorted(set(float(r["rho"]) for r in srows))
        y = [statistics.mean(float(r["hdfs_family_rare"]) for r in srows if float(r["rho"]) == rh) for rh in rhos]
        ax[0].plot(rhos, y, marker="o", ms=3, label=sc, color=SCOL.get(sc), marker="o", ms=3)
    ax[0].set_xlabel(r"$\rho$ (ascension penalty)"); ax[0].set_ylabel("rare-family hit-rate")
    ax[0].legend(fontsize=5.5, frameon=False); ax[0].set_title("a   Scorer x rho (norm=max)", loc="left", fontsize=8)
    ax[0].axvline(0.95, color="#9AA3AB", lw=0.6, ls="--")

    # ssb_r1 vs rho per scorer
    for sc in scorers:
        srows = sorted((r for r in sub if r["scorer"] == sc), key=lambda r: float(r["rho"]))
        rhos = sorted(set(float(r["rho"]) for r in srows))
        y = [statistics.mean(float(r["ssb_r1"]) for r in srows if float(r["rho"]) == rh) for rh in rhos]
        ax[1].plot(rhos, y, marker="s", ms=3, label=sc, color=SCOL.get(sc), marker="o", ms=3)
    ax[1].set_xlabel(r"$\rho$"); ax[1].set_ylabel("SSB rank-1"); ax[1].set_ylim(0.7, 1.02)
    ax[1].axvline(0.95, color="#9AA3AB", lw=0.6, ls="--")
    ax[1].set_title("b   SSB rank-1 (frozen rho=0.95)", loc="left", fontsize=8)
    fig.tight_layout(w_pad=1.8)
    return _save(fig, "ed3_ablation")


if __name__ == "__main__":
    for fn in (figure2, figure4_data, ed1_ssb, ed3_ablation):
        try:
            p = fn()
            print(f"wrote {p}" if p else f"{fn.__name__}: no data, skipped")
        except Exception as e:
            print(f"{fn.__name__}: ERROR {type(e).__name__}: {e}")
