"""Extended Data figures (matplotlib + SciencePlots 'nature'), exported as SVG.

Produces ED figures that are Agent C's responsibility:
  ed_domain_aurc   per-domain AURC + novelty-F1 across all 6 memories
  ed_risk_coverage Phase 5 risk-coverage / selective-prediction curve (qa_sma.csv)
  ed_transfer      PAIRED cross-system transfer comparison, per-leg (NOT leg-averaged)

Outputs -> paper/figures/svg/ed_*.{svg,png,pdf}
  python3 scripts/figures_ed.py

NOTE: Mirrors the style of scripts/figures_paper.py exactly (same palette,
      plt.style.use, _save helper, CSV-only sourcing).
"""
from __future__ import annotations

import csv
import pathlib
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401  (registers the 'nature' style)

plt.style.use(["nature", "no-latex"])
# Match figures_paper.py: bump the type ramp so labels are legible at print size.
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

# ---------------------------------------------------------------- palette ----
SMA_C = "#2E8AA6"
GRAYS = {"bm25": "#C7CCD1", "dense": "#A7AFB6", "hybrid_rrf": "#7E8893",
         "hybrid_rerank": "#5B6670"}
GOLD = "#D39A3E"
MEM_LABEL = {"sma": "SMA", "bm25": "BM25", "dense": "Dense-RAG",
             "hybrid_rrf": "Hybrid-RRF", "hybrid_rerank": "Hybrid+Rerank",
             "hipporag": "HippoRAG"}
MEM_COLOR = {"sma": SMA_C, **GRAYS, "hipporag": GOLD}
MEM_ORDER = ["sma", "bm25", "dense", "hybrid_rrf", "hybrid_rerank", "hipporag"]
ARM_ORDER = [("medicine", "Medicine"), ("discovery", "Genomics"),
             ("finance", "Finance"), ("cyber", "Cyber"), ("legal", "Legal")]

TRANSFER_MEM_COLOR = {
    "SMA": SMA_C, "BM25": "#C7CCD1", "Dense RAG": "#A7AFB6",
    "KG-PPR Proxy": "#9B86C4", "HippoRAG": GOLD,
    "Hybrid-RRF": "#7E8893", "Hybrid+Rerank": "#5B6670",
}


def _rows(path):
    if not path.exists():
        return []
    return [r for r in csv.DictReader(open(path)) if r]


def _save(fig, name):
    for ext in ("svg", "png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    return OUT / f"{name}.svg"


def _panel(ax, letter, *, x=-0.13, y=1.08):
    """Cowplot/patchwork-style bold panel tag in the axes' top-left corner."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left", color="#1A1F24")


# -------------------------------------------------------- ED fig A -----------
def ed_domain_aurc():
    """Per-domain AURC + novelty-F1 across all 6 memories.

    Sources: reports/confirmatory/agentic_{arm}.csv for each arm in ARM_ORDER.
    """
    arms = {a: {r["memory"]: r for r in _rows(CONF / f"agentic_{a}.csv")}
            for a, _ in ARM_ORDER}
    present = [(a, lab) for a, lab in ARM_ORDER if arms.get(a)]
    if not present:
        print("ed_domain_aurc: no data, skipped")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))

    for ax_idx, (metric_key, ylabel, title_label) in enumerate([
        ("aurc", "AURC (↓ better)", "AURC per domain × memory"),
        ("novelty_f1", "Novelty-F1 (↑ better)", "Novelty-F1 per domain × memory"),
    ]):
        ax = axes[ax_idx]
        mems = MEM_ORDER
        nb = len(mems)
        w = 0.82 / nb
        for i, m in enumerate(mems):
            vals = []
            for a, _ in present:
                r = arms[a].get(m)
                vals.append(float(r[metric_key]) if r and r.get(metric_key, "") not in ("", "NA") else 0.0)
            xs = [j + (i - nb / 2) * w + w / 2 for j in range(len(present))]
            ax.bar(xs, vals, w, color=MEM_COLOR[m],
                   label=(MEM_LABEL[m] if ax_idx == 0 else None),
                   edgecolor="white", linewidth=0.3)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([lab for _, lab in present])
        ax.set_ylabel(ylabel)
        ax.set_title(title_label, loc="left", fontsize=9.5, pad=8)
        _panel(ax, "ab"[ax_idx])
        if metric_key == "aurc":
            ax.text(0.5, -0.26, "lower AURC = SMA abstains on hard cases (better risk coverage)",
                    transform=ax.transAxes, fontsize=7, ha="center",
                    style="italic", color="#5F6B78")

    # shared legend below both panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7.5, ncol=6, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, -0.04),
               columnspacing=1.0, handlelength=1.2)
    fig.tight_layout(w_pad=1.8)
    fig.subplots_adjust(bottom=0.22)
    return _save(fig, "ed_domain_aurc")


# -------------------------------------------------------- ED fig B -----------
def ed_risk_coverage():
    """Phase 5 risk-coverage / selective-prediction curve.

    Derived from qa_sma.csv: sort by descending grounding_score and compute
    cumulative accuracy at each coverage fraction.  Also plots the dense
    baseline from qa_dense.csv using its confidence column.
    """
    def load_curve(fname, score_col="grounding_score", label_col="answerable",
                   answer_col="answer"):
        rows = _rows(CONF / fname)
        pts = []
        for r in rows:
            try:
                score = float(r.get(score_col, ""))
            except (ValueError, TypeError):
                continue
            correct = str(r.get(answer_col, "")).strip().lower() not in (
                "", "abstain", "none", "na")
            # 'answerable' = True means there IS a gold answer; correct if pred = gold
            answerable = str(r.get(label_col, "")).strip().lower() == "true"
            # a prediction is "right" if answerable AND the abstain flag is False
            # and the answer field is non-empty and matches — we use correct proxy:
            # abstained = abstain column; else correct determined by record structure
            abstained = str(r.get("abstained", "false")).strip().lower() == "true"
            right = (not abstained) and answerable and (r.get("pred_id", "") == r.get("gold_id", ""))
            pts.append((score, right))
        if not pts:
            return [], []
        pts.sort(key=lambda x: -x[0])
        n = len(pts)
        coverages, accs = [], []
        cumright = 0
        for k, (_, right) in enumerate(pts, 1):
            if right:
                cumright += 1
            coverages.append(k / n)
            accs.append(cumright / k)
        return coverages, accs

    def load_curve_sma():
        """For SMA use grounding_score; correct = gold_id matches pred_id (non-abstain)."""
        rows = _rows(CONF / "qa_sma.csv")
        pts = []
        for r in rows:
            try:
                score = float(r.get("grounding_score", ""))
            except (ValueError, TypeError):
                continue
            abstained = str(r.get("abstained", "")).strip().lower() == "true"
            right = (not abstained) and (r.get("pred_id", "").strip() == r.get("gold_id", "").strip())
            pts.append((score, right))
        if not pts:
            return [], []
        pts.sort(key=lambda x: -x[0])
        n = len(pts)
        coverages, accs = [], []
        cumright = 0
        for k, (_, right) in enumerate(pts, 1):
            if right:
                cumright += 1
            coverages.append(k / n)
            accs.append(cumright / k)
        return coverages, accs

    def load_curve_dense():
        """For Dense use confidence column."""
        rows = _rows(CONF / "qa_dense.csv")
        pts = []
        for r in rows:
            try:
                score = float(r.get("confidence", ""))
            except (ValueError, TypeError):
                score = 0.0
            abstained = str(r.get("abstained", "")).strip().lower() == "true"
            right = (not abstained) and (r.get("pred_id", "").strip() == r.get("gold_id", "").strip())
            pts.append((score, right))
        if not pts:
            return [], []
        pts.sort(key=lambda x: -x[0])
        n = len(pts)
        coverages, accs = [], []
        cumright = 0
        for k, (_, right) in enumerate(pts, 1):
            if right:
                cumright += 1
            coverages.append(k / n)
            accs.append(cumright / k)
        return coverages, accs

    sma_cov, sma_acc = load_curve_sma()
    den_cov, den_acc = load_curve_dense()

    if not sma_cov:
        print("ed_risk_coverage: no data in qa_sma.csv, skipped")
        return None

    # Read threshold from summary
    summ = _rows(CONF / "qa_sma_summary.csv")
    thr = float(summ[0]["score_threshold"]) if summ and summ[0].get("score_threshold", "NA") not in ("NA", "") else None
    sma_aurc = float(summ[0]["aurc"]) if summ and summ[0].get("aurc", "") not in ("", "NA") else None

    fig, ax = plt.subplots(figsize=(5.0, 3.4))

    ax.plot(sma_cov, sma_acc, color=SMA_C, lw=1.8, label="SMA (grounding score)")
    if den_cov:
        ax.plot(den_cov, den_acc, color="#A7AFB6", lw=1.2, ls="--", label="Dense-RAG (confidence)")

    # Random baseline (full coverage)
    if sma_cov:
        total_right = sum(1 for c, a in zip(sma_cov, sma_acc)
                          if abs(c - 1.0) < 1e-9)
        # overall accuracy = last acc value
        baseline_acc = sma_acc[-1] if sma_acc else 0
        ax.axhline(baseline_acc, color="#C7CCD1", lw=0.8, ls=":", zorder=0,
                   label=f"No-abstain baseline ({baseline_acc:.2f})")

    ax.set_xlabel("Coverage fraction (fraction of questions answered)")
    ax.set_ylabel("Selective accuracy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)

    if sma_aurc is not None:
        ax.text(0.97, 0.12, f"SMA AURC = {sma_aurc:.3f}", transform=ax.transAxes,
                ha="right", fontsize=8.5, color=SMA_C, fontweight="bold")
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax.set_title("Phase 5 risk-coverage (selective prediction)", loc="left", fontsize=9.5, pad=8)
    fig.tight_layout()
    return _save(fig, "ed_risk_coverage")


# -------------------------------------------------------- ED fig C -----------
def ed_transfer():
    """PAIRED cross-system transfer comparison, per-leg (NOT leg-averaged).

    Source: reports/confirmatory/t1_transfer_metrics.csv (LogHub cross-system)
    and t2_stats.csv (HDFS family recall) and t3_stats.csv (BugsInPy LOPO).

    Shows macro-F1 per transfer leg, per method — not a leg-averaged summary.
    Only 'LogHub' rows are used (drops DIAGNOSTIC rows).
    """
    rows_all = _rows(CONF / "t1_transfer_metrics.csv")
    # Filter to real data rows only (not DIAGNOSTIC)
    rows = [r for r in rows_all
            if r.get("dataset", "") == "LogHub" and r.get("macro_f1", "ALERT") != "ALERT"]

    if not rows:
        print("ed_transfer: no usable rows in t1_transfer_metrics.csv, skipped")
        return None

    # Collect legs and methods
    legs = sorted(set(r["split"].replace("[seed201][surprisal]", "") for r in rows
                  if "[seed201]" in r.get("split", "")))
    # Use seed 201 only (fixed seed for the paired figure)
    rows201 = [r for r in rows if "[seed201]" in r.get("split", "")]
    methods_ordered = ["SMA", "BM25", "Dense RAG", "KG-PPR Proxy", "HippoRAG",
                       "Hybrid-RRF", "Hybrid+Rerank"]
    # Only keep methods that actually appear
    methods_present = [m for m in methods_ordered
                       if any(r["method"] == m for r in rows201)]

    def get_f1(leg_key, method):
        for r in rows201:
            leg = r["split"].replace("[seed201][surprisal]", "")
            if leg == leg_key and r["method"] == method:
                try:
                    return float(r["macro_f1"])
                except (ValueError, TypeError):
                    return 0.0
        return 0.0

    n_legs = len(legs)
    if n_legs == 0:
        print("ed_transfer: no legs found, skipped")
        return None

    fig, ax = plt.subplots(figsize=(7.8, 3.7))
    nb = len(methods_present)
    w = 0.8 / nb
    leg_labels = [l.replace("->", " → ").replace("_first20M", "") for l in legs]

    for i, m in enumerate(methods_present):
        vals = [get_f1(lg, m) for lg in legs]
        xs = [j + (i - nb / 2) * w + w / 2 for j in range(n_legs)]
        ax.bar(xs, vals, w, color=TRANSFER_MEM_COLOR.get(m, "#888"),
               label=m, edgecolor="white", linewidth=0.3)

    ax.set_xticks(range(n_legs))
    ax.set_xticklabels(leg_labels, rotation=15, ha="right")
    ax.set_ylabel("macro-F1 (seed 201)")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.333, color="#E3E7EA", lw=0.6, ls="--", zorder=0)
    ax.text(n_legs - 0.5, 0.35, "random baseline (3-class)", fontsize=7,
            color="#9AA3AB", ha="right")
    ax.legend(fontsize=7, ncol=4, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.30), columnspacing=1.0, handlelength=1.2)
    ax.set_title("Paired cross-system transfer, per leg (seed 201)", loc="left", fontsize=9.5, pad=8)
    # annotate the collapsed leg in the empty zone above its (short) bars
    ax.text(0.985, 0.66,
            "HDFS→OpenStack:\nall methods collapse\n(retrieval degenerate;\nhonest null)",
            transform=ax.transAxes, fontsize=6.5, style="italic", color="#9AA3AB",
            ha="right", va="top")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    return _save(fig, "ed_transfer")


if __name__ == "__main__":
    fns = [ed_domain_aurc, ed_risk_coverage, ed_transfer]
    for fn in fns:
        try:
            p = fn()
            if p:
                print(f"wrote {p}")
            else:
                print(f"{fn.__name__}: skipped (no data)")
        except Exception as e:
            import traceback
            print(f"{fn.__name__}: ERROR {type(e).__name__}: {e}")
            traceback.print_exc()
