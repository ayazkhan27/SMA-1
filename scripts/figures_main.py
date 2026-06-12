"""Main-text figures for the NMI manuscript (Figure 1 + supplementaries).

Nature spec: double column 183 mm (7.205 in), Arial, vector PDF, colorblind-
safe palette (Okabe-Ito anchors + role colors from paper/figure_specs/README).
Panel E reads ONLY confirmatory CSVs (reports/confirmatory/).
"""

from __future__ import annotations

import csv
import datetime
import pathlib
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
ROOT = pathlib.Path(__file__).resolve().parents[1]
FIGS = ROOT / "paper" / "figures"
CONF = ROOT / "reports" / "confirmatory"

from figstyle import (ENT, FO, HO, INF, VIO, SMA_C, KG_C, METHOD_C, RIBBON,
                      FILL_TEAL, FILL_GREEN, FILL_LAV, FILL_AMBER, FILL_GREY,
                      TITLE_GRAY, panel_title as _ptitle, axis_title, soften_spines)


def rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, cwd=ROOT).stdout.strip()
    except Exception:
        return "?"


def box(ax, x, y, w, h, text, fc, ec, fs=5.2, tc="black", ls="-", lw=0.7, z=3,
        weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=ec, lw=lw, linestyle=ls, zorder=z,
                                mutation_scale=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, zorder=z + 1, weight=weight, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, c="0.25", ls="-", lw=0.8, z=2, ms=5):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=ms, color=c, lw=lw,
                                 linestyle=ls, zorder=z, shrinkA=1, shrinkB=1))


def node(ax, x, y, text, fc, r=0.022, fs=4.6, tc="white", ec=None, ls="-", z=4,
         aspect=1.2):
    # aspect corrects for non-square panels so nodes stay visually round:
    # height_frac = width_frac * (panel_width_in / panel_height_in).
    ax.add_patch(Ellipse((x, y), r * 2.4, r * 2 * aspect, fc=fc, ec=ec or fc,
                         lw=0.8, linestyle=ls, zorder=z))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc, zorder=z + 1)


def panel_label(ax, letter):
    ax.text(0.005, 0.992, letter, transform=ax.transAxes, fontsize=12,
            weight="bold", color=TITLE_GRAY, va="top", ha="left")


def blank(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


# --------------------------------------------------------------- panel A ----
def panel_a(ax):
    blank(ax)
    panel_label(ax, "A")
    ax.text(0.54, 0.992, "Representation: structure, not text", fontsize=7.0,
            ha="center", va="top", weight="bold", color=TITLE_GRAY)
    # raw log excerpt (depicted data), top-left
    ax.add_patch(FancyBboxPatch((0.03, 0.70), 0.45, 0.20, boxstyle="round,pad=0.012",
                                fc="#f4f6f8", ec="0.6", lw=0.6))
    lines = ["1117841440 R63-M0 ciod: timeout on tree link",
             "1117841452 R63-M0 ciod: retry tree link",
             "1117841498 R63-M0 kernel panic"]
    for i, ln in enumerate(lines):
        ax.text(0.05, 0.855 - i * 0.052, ln, fontsize=3.7, family="monospace",
                color="0.25")
    ax.text(0.255, 0.925, "raw session (BGL)", fontsize=4.6, ha="center", color="0.4")
    # encoder arrow into the DAG region
    arrow(ax, 0.255, 0.685, 0.255, 0.585, c="0.2")
    ax.text(0.285, 0.638, "Tier-0 encoder (deterministic rules)", fontsize=4.6,
            color="0.25", va="center")
    # order axis
    for o, y in ((0, 0.20), (1, 0.37), (2, 0.54)):
        ax.text(0.065, y, str(o), fontsize=4.5, color="0.45", ha="center", zorder=6)
    ax.annotate("", xy=(0.030, 0.59), xytext=(0.030, 0.16),
                arrowprops=dict(arrowstyle="->", lw=0.5, color="0.45"))
    ax.text(0.012, 0.38, "order", fontsize=4.5, color="0.45", rotation=90,
            va="center")
    # DAG (entities order 0, first-order 1, higher-order 2)
    e1, e2 = (0.33, 0.20), (0.62, 0.20)
    f1, f2, f3 = (0.27, 0.37), (0.55, 0.37), (0.83, 0.37)
    h1, h2 = (0.41, 0.54), (0.69, 0.54)
    node(ax, *e1, "R63-M0", ENT, r=0.036, fs=4.0)
    node(ax, *e2, "treeLink", ENT, r=0.036, fs=4.0)
    node(ax, *f1, "timeout", FO, r=0.038, fs=4.2)
    node(ax, *f2, "retry", FO, r=0.038, fs=4.2)
    node(ax, *f3, "failure", FO, r=0.038, fs=4.2)
    node(ax, *h1, "cause", HO, r=0.040, fs=4.4)
    node(ax, *h2, "cause", HO, r=0.040, fs=4.4)
    for (hx, hy), kids in ((h1, (f1, f2)), (h2, (f2, f3))):
        for i, (kx, ky) in enumerate(kids):
            arrow(ax, hx, hy - 0.035, kx, ky + 0.04, c=HO, lw=0.6, ms=3.5)
            ax.text((hx + kx) / 2 + 0.015, (hy + ky) / 2 + 0.01, str(i + 1),
                    fontsize=3.6, color="0.45")
    for (fx, fy), kids in ((f1, (e1, e2)), (f2, (e1, e2)), (f3, (e1,))):
        for kx, ky in kids:
            arrow(ax, fx, fy - 0.035, kx, ky + 0.038, c=FO, lw=0.5, ms=3)
    ax.text(0.97, 0.250, "shared sub-expressions hash-consed:\nthe case is a DAG",
            fontsize=4.0, color="0.4", ha="right", style="italic")
    box(ax, 0.06, 0.015, 0.88, 0.115,
        "(cause (timeout R63-M0 treeLink) (retry R63-M0 treeLink))   "
        "case_id = BLAKE3(canonical) = 6c1f6a…", "#eef3f5", SMA_C, fs=3.9)


# --------------------------------------------------------------- panel B ----
def panel_b(ax):
    blank(ax)
    panel_label(ax, "B")
    ax.text(0.54, 0.992, "Architecture: write path, certified read path, one gated LLM",
            fontsize=7.0, ha="center", va="top", weight="bold", color=TITLE_GRAY)
    # WRITE lane
    ax.text(0.015, 0.80, "WRITE", fontsize=5, color="0.45", rotation=90, va="center")
    box(ax, 0.04, 0.70, 0.13, 0.20, "artifacts\nlogs · code\ntraces · obs", "white", "0.4", fs=4.4)
    box(ax, 0.21, 0.70, 0.15, 0.20, "encoders\n(rules; versioned;\nbyte-deterministic)", FILL_TEAL, FO, fs=4.4)
    # store cylinder
    ax.add_patch(FancyBboxPatch((0.41, 0.70), 0.16, 0.17, boxstyle="round,pad=0.012",
                                fc="#eef3f5", ec=SMA_C, lw=0.8))
    ax.add_patch(Ellipse((0.49, 0.875), 0.16, 0.045, fc="#eef3f5", ec=SMA_C, lw=0.8))
    ax.text(0.49, 0.785, "case store\nBLAKE3 ids + WAL", fontsize=4.4, ha="center", va="center")
    box(ax, 0.62, 0.70, 0.16, 0.20, "index\nfunctor postings\n+ WL-1 vectors", FILL_TEAL, FO, fs=4.4)
    for x0, x1 in ((0.17, 0.21), (0.36, 0.41), (0.575, 0.62)):
        arrow(ax, x0, 0.80, x1, 0.80)
    # READ lane
    ax.text(0.015, 0.38, "READ", fontsize=5, color="0.45", rotation=90, va="center")
    box(ax, 0.04, 0.28, 0.11, 0.20, "query\nartifact", "white", "0.4", fs=4.4)
    box(ax, 0.19, 0.28, 0.17, 0.20, "MAC: admissible\nbound orders\ncandidates", FILL_TEAL, FO, fs=4.4)
    box(ax, 0.40, 0.28, 0.17, 0.20, "FAC: SME align,\nbest-first,\ncertified top-k", "#d3e2e8", HO, fs=4.4, tc="black")
    box(ax, 0.61, 0.28, 0.15, 0.20, "receipts:\nmaps · scores ·\ninferences", "white", INF, fs=4.4)
    for x0, x1 in ((0.15, 0.19), (0.36, 0.40), (0.57, 0.61)):
        arrow(ax, x0, 0.38, x1, 0.38)
    # MAC glyph: overlapping histograms
    for i, (c, dx) in enumerate(((FO, 0.0), ("0.6", 0.012))):
        for j, hgt in enumerate((0.030, 0.052, 0.022)):
            ax.add_patch(plt.Rectangle((0.215 + j * 0.034 + dx, 0.295), 0.011, hgt,
                                       fc=c, alpha=0.75, zorder=5))
    # LLM gate
    ax.add_patch(FancyBboxPatch((0.80, 0.24), 0.185, 0.30, boxstyle="round,pad=0.012",
                                fc="none", ec="0.35", lw=0.7, linestyle="--"))
    box(ax, 0.815, 0.30, 0.155, 0.18, "LLM verbalizes\nreceipts only:\nCITE or ABSTAIN", FILL_AMBER, INF, fs=4.4)
    ax.text(0.8925, 0.56, "the only LLM — never writes facts", fontsize=4.2,
            ha="center", color="0.35", style="italic")
    arrow(ax, 0.76, 0.38, 0.815, 0.38)
    arrow(ax, 0.49, 0.70, 0.49, 0.50, c="0.45", ls=":")  # store feeds read lane
    ax.text(0.505, 0.60, "single source of truth", fontsize=4.0, color="0.45")
    # legend
    items = [("entity", ENT), ("1st-order rel.", FO), ("higher-order rel.", HO),
             ("candidate inference", INF), ("lattice ascension", VIO)]
    x = 0.045
    for lbl, c in items:
        ax.add_patch(plt.Rectangle((x, 0.055), 0.022, 0.05, fc=c))
        ax.text(x + 0.028, 0.08, lbl, fontsize=4.3, va="center")
        x += 0.028 + 0.022 + len(lbl) * 0.0062
    ax.text(0.045, 0.005, "solid = deterministic flow    dashed = hypothetical    dotted = derived/ascension",
            fontsize=4.0, color="0.45")


# --------------------------------------------------------------- panel C ----
C_ASPECT = 3.9  # panel C is ~5:1 wide; keep its nodes visually round


def _mini_dag(ax, cx, cy, col, s=0.023):
    """3-node mini DAG (HO over two FO); returns node coords."""
    top = (cx, cy + s * 4.6)
    l, r = (cx - s, cy - s * 3.4), (cx + s, cy - s * 3.4)
    for k in (l, r):
        arrow(ax, top[0], top[1] - 0.05, k[0], k[1] + 0.055, c=col, lw=0.5, ms=3)
    node(ax, *top, "", col, r=0.0115, aspect=C_ASPECT)
    node(ax, *l, "", FO if col == HO else col, r=0.0095, aspect=C_ASPECT)
    node(ax, *r, "", FO if col == HO else col, r=0.0095, aspect=C_ASPECT)
    return top, l, r


def panel_c(ax):
    blank(ax)
    panel_label(ax, "C")
    ax.text(0.54, 0.99, "Inside the matcher: exact-anytime structure mapping (SME core)",
            fontsize=7.0, ha="center", va="top", weight="bold", color=TITLE_GRAY)
    y = 0.56
    stages = [0.115, 0.30, 0.50, 0.70, 0.885]
    # 1 seeding
    b = _mini_dag(ax, stages[0] - 0.050, y, HO)
    t = _mini_dag(ax, stages[0] + 0.050, y, HO)
    for p, q in zip(b, t):
        ax.plot([p[0] + 0.013, q[0] - 0.013], [p[1], q[1]], ls=":", lw=0.6,
                color="0.4", zorder=1)
    # 2 kernels
    b = _mini_dag(ax, stages[1] - 0.050, y, HO)
    t = _mini_dag(ax, stages[1] + 0.050, y, HO)
    ax.add_patch(Ellipse((stages[1], y + 0.01), 0.155, 0.50, fc=SMA_C, alpha=0.09,
                         ec=SMA_C, lw=0.6, zorder=0))
    for p, q in zip(b, t):
        ax.plot([p[0] + 0.013, q[0] - 0.013], [p[1], q[1]], ls=":", lw=0.6,
                color="0.4", zorder=1)
    # 3 conflict graph -> MWIS (k1,k4 selected; k2,k3 conflicted out)
    kpos = [(stages[2] - 0.045, y + 0.16), (stages[2] + 0.045, y + 0.16),
            (stages[2] - 0.045, y - 0.16), (stages[2] + 0.045, y - 0.16)]
    for i, j in ((0, 1), (1, 2)):
        ax.plot([kpos[i][0], kpos[j][0]], [kpos[i][1], kpos[j][1]], color=INF,
                lw=0.8, zorder=1)
    for i, (kx, ky) in enumerate(kpos):
        node(ax, kx, ky, f"k{i+1}", SMA_C if i in (0, 3) else "0.75", r=0.013,
             fs=3.8, aspect=C_ASPECT)
        if i in (0, 3):
            ax.add_patch(Ellipse((kx, ky), 0.046, 0.046 * C_ASPECT, fc="none",
                                 ec=HO, lw=0.9))
    # 4 trickle-down
    top, l, r = _mini_dag(ax, stages[3] - 0.02, y, HO, s=0.05)
    for (px, py), w in ((top, 0.042), (l, 0.024), (r, 0.024)):
        ax.add_patch(plt.Rectangle((px + 0.015, py - 0.012), w, 0.024, fc=VIO,
                                   alpha=0.9, zorder=5))
    ax.text(stages[3], y + 0.30, r"s(h) = $\sigma_0$·asc + $\gamma\,\Sigma$ s(parents)",
            fontsize=4.2, ha="center", color="0.35")
    # 5 candidate inference
    b = _mini_dag(ax, stages[4] - 0.050, y, HO)
    node(ax, stages[4] - 0.050, y - 0.31, "", FO, r=0.0095, aspect=C_ASPECT)
    arrow(ax, stages[4] - 0.050, y - 0.16, stages[4] - 0.050, y - 0.25, c=FO,
          lw=0.5, ms=3)
    t = _mini_dag(ax, stages[4] + 0.050, y, HO)
    node(ax, stages[4] + 0.050, y - 0.31, "?", "white", r=0.012, fs=4.5,
         tc=INF, ec=INF, ls="--", aspect=C_ASPECT)
    arrow(ax, stages[4] - 0.036, y - 0.31, stages[4] + 0.033, y - 0.31, c=INF,
          ls="--", lw=0.7, ms=4)
    labels = ["1  seed match\nhypotheses", "2  close support\n→ kernels",
              "3  conflicts → MWIS\n(CP-SAT, gap logged)", "4  systematicity\nscoring",
              "5  project inference\n(:hypothetical)"]
    for x, lbl in zip(stages, labels):
        ax.text(x, 0.10, lbl, fontsize=4.6, ha="center", va="top", color="0.25")
    for x0, x1 in zip(stages[:-1], stages[1:]):
        ax.text((x0 + x1) / 2, y, "▸", fontsize=8, color="0.45",
                ha="center", va="center", family="DejaVu Sans")


# --------------------------------------------------------------- panel D ----
TASKS = [("SSB (synthetic gold)", "queued"), ("BGL→Spirit", "done"),
         ("BGL→Thunderbird", "done"), ("HDFS→OpenStack", "done"),
         ("HDFS failure families", "running"), ("BGL triage", "running"),
         ("BugsInPy (code)", "queued"), ("Liberty haystack", "queued")]
METHODS = ["SMA", "BM25", "Dense", "Hyb-RRF", "Hyb+RR", "KG-PPR", "HippoRAG"]
COVER = {  # which methods run per task (battery design)
    "SSB (synthetic gold)": {"SMA", "BM25", "Dense"},
    "BGL→Spirit": set(METHODS), "BGL→Thunderbird": set(METHODS),
    "HDFS→OpenStack": set(METHODS), "BGL triage": set(METHODS),
    "HDFS failure families": {"SMA", "BM25", "Dense"},
    "BugsInPy (code)": {"SMA", "BM25", "Dense"},
    "Liberty haystack": {"SMA", "BM25", "Dense", "Hyb-RRF"},
}


def panel_d(ax):
    blank(ax)
    panel_label(ax, "D")
    ax.text(0.56, 0.992, "Pre-registered battery: tasks × methods", fontsize=7.0,
            ha="center", va="top", weight="bold", color=TITLE_GRAY)
    from matplotlib.patches import Wedge

    x0, y0, dx, dy = 0.30, 0.745, 0.082, 0.082

    def status_dot(xx, yy, status):
        r = 0.013
        if status == "done":
            ax.add_patch(Ellipse((xx, yy), r * 2, r * 2.6, fc=SMA_C, ec=SMA_C, lw=0.6))
        elif status == "running":
            ax.add_patch(Ellipse((xx, yy), r * 2, r * 2.6, fc="white", ec="#b97c0a", lw=0.7))
            ax.add_patch(Wedge((xx, yy), r, 90, 270, fc="#b97c0a", ec="none"))
        else:
            ax.add_patch(Ellipse((xx, yy), r * 2, r * 2.6, fc="white", ec="0.55", lw=0.7))

    # MAMMAL-style colored stage ribbon over the method groups
    groups = [("structural", [0], RIBBON[0]), ("lexical", [1, 2], RIBBON[1]),
              ("hybrid", [3, 4], RIBBON[2]), ("graph", [5, 6], RIBBON[3])]
    rib_y = y0 + 0.115
    for name, cols, col in groups:
        xa = x0 + (min(cols) - 0.42) * dx
        xb = x0 + (max(cols) + 0.42) * dx
        ax.add_patch(FancyBboxPatch((xa, rib_y), xb - xa, 0.038,
                                    boxstyle="round,pad=0.004", fc=col, ec="none",
                                    zorder=2))
        ax.text((xa + xb) / 2, rib_y + 0.019, name, fontsize=3.9, ha="center",
                va="center", color="#3d4751", zorder=3)
        # faint group column tint down the matrix
        ax.add_patch(plt.Rectangle((xa, y0 - dy * 7.45), xb - xa, dy * 7.95,
                                   fc=col, alpha=0.18, zorder=0))
    for j, m in enumerate(METHODS):
        ax.text(x0 + j * dx, y0 + 0.055, m, fontsize=4.2, ha="left", rotation=38,
                color=SMA_C if m == "SMA" else "0.3", rotation_mode="anchor",
                weight="bold" if m == "SMA" else "normal")
    for i, (task, status) in enumerate(TASKS):
        yy = y0 - i * dy
        ax.text(0.275, yy, task, fontsize=4.4, ha="right", va="center", color="0.25")
        for j, m in enumerate(METHODS):
            if m in COVER[task]:
                ax.text(x0 + j * dx, yy, "✓", fontsize=5.0, ha="center",
                        va="center", color=SMA_C if m == "SMA" else "0.45",
                        family="DejaVu Sans")
        status_dot(x0 + 6.9 * dx, yy, status)
    ax.text(x0 + 6.9 * dx, y0 + 0.055, "status", fontsize=4.2, ha="left",
            rotation=38, color="0.3", rotation_mode="anchor")
    # status legend (drawn, not unicode)
    status_dot(0.06, 0.085, "done")
    ax.text(0.085, 0.085, "confirmatory complete", fontsize=4.2, va="center", color="0.4")
    status_dot(0.40, 0.085, "running")
    ax.text(0.425, 0.085, "running", fontsize=4.2, va="center", color="0.4")
    status_dot(0.57, 0.085, "queued")
    ax.text(0.595, 0.085, "queued", fontsize=4.2, va="center", color="0.4")
    ax.text(0.5, 0.030, "seeds 201–205 / 41,43, frozen at prereg-v1 · deterministic extraction ($0 LLM tokens)\n"
            "statistics: paired bootstrap + Holm–Bonferroni + Cliff's δ",
            fontsize=3.9, ha="center", va="top", color="0.4")


# --------------------------------------------------------------- panel E ----
def load_t1():
    rows = list(csv.DictReader((CONF / "t1_summary.csv").open()))
    stats = list(csv.DictReader((CONF / "t1_stats.csv").open()))
    return rows, stats


LEGS = [("BGL->spirit_first20M", "BGL→Spirit"),
        ("BGL->thunderbird_first20M", "BGL→Thunderbird"),
        ("HDFS->OpenStack", "HDFS→OpenStack")]
ORDER = ["SMA", "Hybrid+Rerank", "Hybrid-RRF", "HippoRAG", "KG-PPR Proxy",
         "Dense RAG", "BM25"]
SHORT = {"SMA": "SMA", "Hybrid+Rerank": "Hyb+RR", "Hybrid-RRF": "Hyb-RRF",
         "HippoRAG": "HippoRAG", "KG-PPR Proxy": "KG-PPR", "Dense RAG": "Dense",
         "BM25": "BM25"}


def panel_e(ax):
    rows, stats = load_t1()
    ax.set_title("Confirmatory cross-system transfer (T1): label-hit@1, 5 seeds",
                 fontsize=6.8, weight="bold", color=TITLE_GRAY, pad=3)
    width = 0.105
    for li, (leg, leg_lbl) in enumerate(LEGS):
        for mi, m in enumerate(ORDER):
            r = next(r for r in rows if r["leg"] == leg and r["method"] == m
                     and r["metric"] == "hit@1")
            x = li + (mi - 3) * width
            ax.bar(x, float(r["mean"]), width * 0.92, yerr=float(r["sd"]),
                   color=METHOD_C[m], error_kw=dict(lw=0.7, capsize=1.5,
                   capthick=0.7), edgecolor="black" if m == "SMA" else "none",
                   linewidth=0.7, zorder=3)
        # significance annotation vs best baseline
        sm = next(r for r in rows if r["leg"] == leg and r["method"] == "SMA"
                  and r["metric"] == "hit@1")
        base_rows = [r for r in rows if r["leg"] == leg and r["method"] != "SMA"
                     and r["metric"] == "hit@1"]
        best = max(base_rows, key=lambda r: float(r["mean"]))
        st = next(s for s in stats if s["leg"] == leg and s["baseline"] == best["method"])
        p = float(st["p_holm"])
        delta = float(st["delta"])
        tag = ("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05
               else "n.s.")
        ann_y = (0.94, 1.06, 0.68)[li]
        ann_x = li + (0.22 if li == 0 else 0.0)
        ax.text(ann_x, ann_y, f"Δ={delta:+.2f} vs {SHORT[best['method']]} ({tag})",
                fontsize=4.6, ha="center", color="0.25")
    ax.axhline(0.5, color="0.6", lw=0.6, ls=":")
    ax.text(-0.72, 0.505, "chance", fontsize=4.4, color="0.5", va="bottom")
    ax.set_xticks(range(3), [l for _, l in LEGS], fontsize=5.5)
    ax.set_xlim(-0.78, 2.45)
    ax.set_ylim(0, 1.14)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.50", "0.75", "1.00"],
                  fontsize=5)
    ax.set_ylabel("label-hit@1", fontsize=6)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=METHOD_C[m],
               ec="black" if m == "SMA" else "none", lw=0.7) for m in ORDER]
    ax.legend(handles, [SHORT[m] for m in ORDER], fontsize=4.4, ncol=1,
              frameon=False, loc="upper left", bbox_to_anchor=(-0.005, 1.02),
              handlelength=1.0, labelspacing=0.35, handletextpad=0.4)
    panel_label(ax, "E")


# ------------------------------------------------------------------ build ----
def figure1():
    fig = plt.figure(figsize=(7.205, 6.4))
    gs = fig.add_gridspec(3, 5, height_ratios=[1.5, 1.05, 1.55],
                          hspace=0.16, wspace=0.55,
                          left=0.01, right=0.995, top=0.995, bottom=0.045)
    panel_a(fig.add_subplot(gs[0, :2]))
    panel_b(fig.add_subplot(gs[0, 2:]))
    panel_c(fig.add_subplot(gs[1, :]))
    panel_d(fig.add_subplot(gs[2, :2]))
    ax_e = fig.add_subplot(gs[2, 2:])
    ax_e.set_position([0.50, 0.065, 0.48, 0.235])
    panel_e(ax_e)
    fig.text(0.995, 0.002, f"{datetime.date.today()} · {rev()} · prereg-v1",
             fontsize=3.5, ha="right", color="0.65")
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig1_sma_overview.{ext}")
    plt.close(fig)
    print("fig1_sma_overview.pdf/.png")


def figure_s1():
    """Supplementary Fig. 1: full T1 statistics (macro-F1 + delta forest)."""
    rows, stats = load_t1()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.205, 2.9),
                                   gridspec_kw=dict(width_ratios=[1.15, 1],
                                                    wspace=0.30, left=0.06,
                                                    right=0.99, top=0.84,
                                                    bottom=0.13))
    width = 0.105
    for li, (leg, leg_lbl) in enumerate(LEGS):
        for mi, m in enumerate(ORDER):
            r = next(r for r in rows if r["leg"] == leg and r["method"] == m
                     and r["metric"] == "macro_f1")
            ax1.bar(li + (mi - 3) * width, float(r["mean"]), width * 0.92,
                    yerr=float(r["sd"]), color=METHOD_C[m],
                    error_kw=dict(lw=0.7, capsize=1.5, capthick=0.7),
                    edgecolor="black" if m == "SMA" else "none", linewidth=0.7,
                    zorder=3)
    ax1.axhline(1 / 3, color="0.6", lw=0.6, ls=":")
    ax1.text(2.42, 0.34, "collapse", fontsize=4.4, color="0.5", va="bottom")
    ax1.set_xticks(range(3), [l for _, l in LEGS], fontsize=5.5)
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("macro-F1", fontsize=6)
    ax1.tick_params(labelsize=5)
    ax1.spines[["top", "right"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=METHOD_C[m],
               ec="black" if m == "SMA" else "none", lw=0.7) for m in ORDER]
    ax1.legend(handles, [SHORT[m] for m in ORDER], fontsize=4.4, ncol=4,
               frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.17),
               handlelength=1.0, columnspacing=0.8, handletextpad=0.4)
    ax1.set_title("a  Transfer macro-F1\n(5 seeds, mean ± s.d.)", fontsize=6.8,
                  weight="bold", loc="left", pad=3, color=TITLE_GRAY)

    # forest of per-query paired-bootstrap deltas (SMA - baseline, hit@1)
    yy = 0
    ylabels, ypos = [], []
    for leg, leg_lbl in LEGS:
        for st in [s2 for s2 in stats if s2["leg"] == leg]:
            d, lo, hi = (float(st["delta"]), float(st["ci_low"]),
                         float(st["ci_high"]))
            p = float(st["p_holm"])
            c = SMA_C if lo > 0 else (INF if hi < 0 else "0.55")
            ax2.errorbar(d, yy, xerr=[[d - lo], [hi - d]], fmt="o", ms=2.2,
                         color=c, elinewidth=0.9, capsize=1.6, capthick=0.8)
            star = ("***" if p < 0.001 else "**" if p < 0.01 else
                    "*" if p < 0.05 else "")
            if star:
                ax2.text(hi + 0.012, yy, star, fontsize=4.6, va="center", color=c)
            ylabels.append(f"{leg_lbl}:  {SHORT[st['baseline']]}")
            ypos.append(yy)
            yy -= 1
        yy -= 0.8
    ax2.axvline(0, color="0.3", lw=0.6)
    ax2.set_yticks(ypos, ylabels, fontsize=4.4)
    ax2.tick_params(labelsize=5)
    ax2.set_xlabel("Δ label-hit@1 (SMA − baseline), 95% CI", fontsize=5.5)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_title("b  Paired-bootstrap deltas (Holm-corrected)", fontsize=6.8,
                  weight="bold", loc="left", pad=4, color=TITLE_GRAY)
    fig.text(0.995, 0.002, f"{datetime.date.today()} · {rev()} · prereg-v1",
             fontsize=3.5, ha="right", color="0.65")
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"figS1_t1_statistics.{ext}")
    plt.close(fig)
    print("figS1_t1_statistics.pdf/.png")


if __name__ == "__main__":
    # Figure 1 is now generated by scripts/figure1_tikz.py (TikZ, MAMMAL-style).
    # This module owns the supplementary statistics figure only.
    figure_s1()
