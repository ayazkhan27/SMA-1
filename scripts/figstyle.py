"""Shared figure style for the SMA manuscript — Nature/MMAL-grade.

Mimics the *style* of npj/Nature multi-panel figures (gray-blue panel titles,
airy pastel category palette, colored stage ribbons, circular glyphs) without
copying any content. Imported by every figure script so the look is uniform.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

# --- typography ------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "figure.dpi": 300, "savefig.dpi": 600,
})

# MAMMAL-style muted gray-blue for panel letters + titles (not pure black).
TITLE_GRAY = "#5f6b78"

# --- role palette (semantic; carried across every figure) ------------------
ENT = "#7b8794"      # entities / constants (soft slate)
FO = "#5aa9c4"       # first-order relations (soft teal)
HO = "#2e6b86"       # higher-order relations (deeper teal, the systematicity carrier)
INF = "#d98a3d"      # candidate inference (warm amber-orange), always dashed
VIO = "#9b86c4"      # lattice ascension (soft violet)
SMA_C = "#2e8aa6"    # SMA method line
KG_C = "#e7b15a"     # KG family (soft gold)

# Soft pastel fills for boxes (MAMMAL uses very light tints).
FILL_TEAL = "#e3eef2"
FILL_GREEN = "#e6f0e6"
FILL_LAV = "#ece8f4"
FILL_AMBER = "#fbf0e2"
FILL_GREY = "#f1f3f5"

# Method palette for results panels (colorblind-safe, soft).
GRAYS = {"BM25": "#c7ccd1", "Dense RAG": "#a7afb6", "Hybrid-RRF": "#7e8893",
         "Hybrid+Rerank": "#5b6670"}
METHOD_C = {"SMA": SMA_C, "KG-PPR Proxy": KG_C, "HippoRAG": "#d39a3e", **GRAYS}

# Pastel stage-ribbon palette (MAMMAL's colored column-header bars).
RIBBON = ["#cfe3d6", "#cfe0e8", "#d7d2e8", "#f3ddd0"]  # green→teal→lav→peach


def panel_title(ax, letter, title, x=0.5, y=0.99, fs_letter=12, fs_title=6.8):
    """MAMMAL-style: gray-blue bold letter + title, top of a blank axis."""
    ax.text(0.004, y, letter, transform=ax.transAxes, fontsize=fs_letter,
            weight="bold", color=TITLE_GRAY, va="top", ha="left")
    ax.text(x, y, title, transform=ax.transAxes, fontsize=fs_title,
            weight="bold", color=TITLE_GRAY, va="top", ha="center")


def axis_title(ax, letter, title, fs_letter=11, fs_title=6.8, pad=4, loc="left"):
    """For data axes: 'a  Title' in gray-blue, top-left."""
    ax.set_title(f"{letter}  {title}", fontsize=fs_title, weight="bold",
                 color=TITLE_GRAY, loc=loc, pad=pad)


def soften_spines(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#9aa3ab")
