"""Generate the SMA-1 Figure 1 composite as standalone TikZ (xelatex/Arial).

MAMMAL-style multi-panel: A representation, B architecture, C matcher,
D coverage matrix with stage ribbon, E confirmatory T1 transfer (from CSV).
Emits paper/figures/tikz/fig1.tex. Compile via the tikz-diagrams skill.
"""
from __future__ import annotations
import csv, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONF = ROOT / "reports" / "confirmatory"
OUT = ROOT / "paper" / "figures" / "tikz" / "fig1.tex"

# ---- T1 data for panel E -------------------------------------------------
LEGS = [("BGL->spirit_first20M", "BGL$\\rightarrow$Spirit"),
        ("BGL->thunderbird_first20M", "BGL$\\rightarrow$Thunderbird"),
        ("HDFS->OpenStack", "HDFS$\\rightarrow$OpenStack")]
BARMETH = ["SMA", "BM25", "Dense RAG", "Hybrid-RRF", "Hybrid+Rerank",
           "KG-PPR Proxy", "HippoRAG"]
BARCOL = {"SMA": "smac", "BM25": "g1", "Dense RAG": "g2", "Hybrid-RRF": "g3",
          "Hybrid+Rerank": "g4", "KG-PPR Proxy": "kgc", "HippoRAG": "kgc2"}


def t1():
    rows = list(csv.DictReader((CONF / "t1_summary.csv").open()))
    stats = list(csv.DictReader((CONF / "t1_stats.csv").open()))
    out = {}
    for leg, _ in LEGS:
        out[leg] = {}
        for m in BARMETH:
            r = next(r for r in rows if r["leg"] == leg and r["method"] == m
                     and r["metric"] == "hit@1")
            out[leg][m] = (float(r["mean"]), float(r["sd"]))
        base = [r for r in rows if r["leg"] == leg and r["method"] != "SMA"
                and r["metric"] == "hit@1"]
        best = max(base, key=lambda r: float(r["mean"]))
        st = next(s for s in stats if s["leg"] == leg
                  and s["baseline"] == best["method"])
        short = {"Hybrid-RRF": "Hyb-RRF", "Hybrid+Rerank": "Hyb+RR",
                 "HippoRAG": "HippoRAG"}.get(best["method"], best["method"])
        out[leg]["_best"] = (short, float(st["delta"]), float(st["p_holm"]))
    return out


# ---- TeX preamble --------------------------------------------------------
PREAMBLE = r"""\documentclass[border=2pt]{standalone}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{fontspec}
\setsansfont{Arial}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,calc,fit,backgrounds,shapes.geometric}
\definecolor{titlegray}{HTML}{5F6B78}
\definecolor{ent}{HTML}{7B8794}
\definecolor{fo}{HTML}{5AA9C4}
\definecolor{ho}{HTML}{2E6B86}
\definecolor{inf}{HTML}{D98A3D}
\definecolor{vio}{HTML}{9B86C4}
\definecolor{smac}{HTML}{2E8AA6}
\definecolor{kgc}{HTML}{E7B15A}
\definecolor{kgc2}{HTML}{D39A3E}
\definecolor{g1}{HTML}{C7CCD1}
\definecolor{g2}{HTML}{A7AFB6}
\definecolor{g3}{HTML}{7E8893}
\definecolor{g4}{HTML}{5B6670}
\definecolor{fillteal}{HTML}{E3EEF2}
\definecolor{fillamber}{HTML}{FBF0E2}
\definecolor{fillgrey}{HTML}{F1F3F5}
\definecolor{ribA}{HTML}{CFE3D6}
\definecolor{ribB}{HTML}{CFE0E8}
\definecolor{ribC}{HTML}{D7D2E8}
\definecolor{ribD}{HTML}{F3DDD0}
\definecolor{inkmid}{HTML}{3D4751}
\begin{document}
\begin{tikzpicture}[
  font=\sffamily,
  >=Stealth,
  ptitle/.style={font=\sffamily\bfseries\small, color=titlegray},
  pletter/.style={font=\sffamily\bfseries, color=titlegray, scale=1.35},
  ttl/.style={font=\sffamily\fontsize{6}{7}\selectfont, color=titlegray, align=center},
  sub/.style={font=\sffamily\fontsize{4.6}{5.4}\selectfont, color=black!55, align=center},
  micro/.style={font=\sffamily\fontsize{4}{4.8}\selectfont, color=black!50, align=center},
  enode/.style={circle, draw=ent, fill=ent, text=white, font=\sffamily\fontsize{4}{4.6}\selectfont, inner sep=0pt, minimum size=4.2mm, align=center},
  fnode/.style={circle, draw=fo, fill=fo, text=white, font=\sffamily\fontsize{4}{4.6}\selectfont, inner sep=0pt, minimum size=4.6mm, align=center},
  hnode/.style={circle, draw=ho, fill=ho, text=white, font=\sffamily\fontsize{4}{4.6}\selectfont, inner sep=0pt, minimum size=5mm, align=center},
  pbox/.style={rounded corners=1.5pt, draw, align=center, inner sep=2.5pt, font=\sffamily\fontsize{4.4}{5.2}\selectfont},
  flowarr/.style={->, line width=0.5pt, color=black!55},
  horel/.style={->, line width=0.5pt, color=ho},
  forel/.style={->, line width=0.4pt, color=fo!80},
]
"""

FOOT = r"""\end{tikzpicture}
\end{document}
"""


def esc(s):
    return s.replace("&", "\\&").replace("_", "\\_")


def build():
    d = t1()
    L = []
    a = L.append

    # ================= PANEL A : representation =================
    a(r"\begin{scope}[shift={(0,11.9)}]")
    a(r"\node[pletter] at (-0.05,4.35) {A};")
    a(r"\node[ttl] at (4.3,4.35) {Representation: structure, not text};")
    # raw log card
    a(r"\node[draw=black!40, fill=fillgrey, rounded corners=2pt, align=left, "
      r"inner sep=3pt, font=\ttfamily\fontsize{3.6}{4.4}\selectfont, text width=4.1cm] "
      r"(log) at (2.15,3.55) {1117841440 R63-M0 ciod: timeout tree link\\"
      r"1117841452 R63-M0 ciod: retry tree link\\1117841498 R63-M0 kernel panic};")
    a(r"\node[micro] at (2.15,4.0) {raw session (BGL)};")
    a(r"\node[micro, align=center] (enc) at (2.15,2.78) {Tier-0 encoder\\(deterministic rules)};")
    a(r"\draw[flowarr] (2.15,3.12) -- (2.15,2.96);")
    a(r"\draw[flowarr] (2.15,2.6) -- (2.15,2.42);")
    # order axis
    a(r"\draw[->, line width=0.4pt, color=black!45] (0.18,0.35) -- (0.18,2.25);")
    a(r"\node[micro, rotate=90] at (0.05,1.3) {order};")
    for o, y in ((0, 0.55), (1, 1.30), (2, 2.05)):
        a(rf"\node[micro] at (0.33,{y}) {{{o}}};")
    # DAG nodes
    a(r"\node[enode] (e1) at (1.5,0.55) {R63-M0};")
    a(r"\node[enode] (e2) at (3.4,0.55) {treeLink};")
    a(r"\node[fnode] (f1) at (1.2,1.30) {timeout};")
    a(r"\node[fnode] (f2) at (2.6,1.30) {retry};")
    a(r"\node[fnode] (f3) at (4.0,1.30) {failure};")
    a(r"\node[hnode] (h1) at (1.9,2.05) {cause};")
    a(r"\node[hnode] (h2) at (3.3,2.05) {cause};")
    for h, ks in (("h1", ("f1", "f2")), ("h2", ("f2", "f3"))):
        for i, k in enumerate(ks, 1):
            a(rf"\draw[horel] ({h}) -- ({k}) node[midway, font=\sffamily\fontsize{{3.2}}{{3.6}}\selectfont, color=black!45, inner sep=1pt] {{{i}}};")
    for f, ks in (("f1", ("e1", "e2")), ("f2", ("e1", "e2")), ("f3", ("e1",))):
        for k in ks:
            a(rf"\draw[forel] ({f}) -- ({k});")
    a(r"\node[micro, align=right, text width=2.6cm] at (6.2,1.4) "
      r"{\textit{shared sub-expressions}\\\textit{hash-consed: the case}\\\textit{is a DAG}};")
    a(r"\node[pbox, draw=smac, fill=fillteal, text=black, font=\sffamily\fontsize{3.8}{4.6}\selectfont, "
      r"text width=8.2cm, align=center] at (4.3,0.12) "
      r"{\texttt{(cause (timeout R63-M0 treeLink) (retry R63-M0 treeLink))}\quad "
      r"case\_id = BLAKE3(canonical) = 6c1f6a\ldots};")
    a(r"\end{scope}")

    # ================= PANEL B : architecture =================
    a(r"\begin{scope}[shift={(9.5,11.9)}]")
    a(r"\node[pletter] at (-0.05,4.35) {B};")
    a(r"\node[ttl] at (4.3,4.35) {Architecture: write path, certified read path, one gated LLM};")
    # write lane
    a(r"\node[micro, rotate=90, color=black!45] at (-0.05,3.4) {WRITE};")
    a(r"\node[pbox, draw=black!40, fill=white, text width=1.5cm, minimum height=8mm] (art) at (0.9,3.4) {artifacts\\logs $\cdot$ code\\traces $\cdot$ obs};")
    a(r"\node[pbox, draw=fo, fill=fillteal, text width=1.8cm, minimum height=8mm] (enc2) at (3.0,3.4) {encoders\\rules; versioned;\\byte-deterministic};")
    a(r"\node[cylinder, shape border rotate=90, draw=smac, fill=fillteal, aspect=0.25, "
      r"minimum width=1.9cm, minimum height=1.0cm, inner sep=1pt, "
      r"font=\sffamily\fontsize{4.2}{5}\selectfont, align=center] (store) at (5.4,3.45) {case store\\BLAKE3 ids\\+ WAL};")
    a(r"\node[pbox, draw=fo, fill=fillteal, text width=1.8cm, minimum height=8mm] (idx) at (7.6,3.4) {index\\functor postings\\+ WL-1 vectors};")
    a(r"\draw[flowarr] (art) -- (enc2); \draw[flowarr] (enc2) -- (store); \draw[flowarr] (store) -- (idx);")
    # read lane
    a(r"\node[micro, rotate=90, color=black!45] at (-0.05,1.25) {READ};")
    a(r"\node[pbox, draw=black!40, fill=white, text width=1.3cm, minimum height=8mm] (q) at (0.85,1.25) {query\\artifact};")
    a(r"\node[pbox, draw=fo, fill=fillteal, text width=1.85cm, minimum height=8mm] (mac) at (3.0,1.25) {MAC: admissible\\bound orders\\candidates};")
    a(r"\node[pbox, draw=ho, fill=ho!18, text width=1.85cm, minimum height=8mm] (fac) at (5.3,1.25) {FAC: SME align,\\best-first,\\certified top-$k$};")
    a(r"\node[pbox, draw=inf, fill=white, text width=1.6cm, minimum height=8mm] (rec) at (7.45,1.25) {receipts:\\maps $\cdot$ scores\\$\cdot$ inferences};")
    a(r"\draw[flowarr] (q) -- (mac); \draw[flowarr] (mac) -- (fac); \draw[flowarr] (fac) -- (rec);")
    a(r"\draw[flowarr, dashed] (store) -- node[micro, right=1pt] {single source of truth} (5.4,1.9);")
    # LLM gate
    a(r"\node[pbox, draw=inf, fill=fillamber, text width=1.7cm, minimum height=9mm] (llm) at (8.55,0.0) {LLM verbalizes\\receipts only:\\CITE or ABSTAIN};")
    a(r"\node[draw=black!35, dashed, rounded corners=2pt, fit=(llm), inner sep=3pt, "
      r"label={[micro, yshift=-1pt]above:the only LLM --- never writes facts}] {};")
    a(r"\draw[flowarr] (rec) |- (llm);")
    # legend
    a(r"\foreach \x/\c/\t in {0.4/ent/entity, 1.9/fo/{1st-order rel.}, 3.6/ho/{higher-order rel.}, 5.5/inf/{candidate inf.}, 7.1/vio/{lattice ascension}} "
      r"{\fill[\c] (\x,-0.62) rectangle ++(0.16,0.16); \node[micro, anchor=west] at (\x+0.20,-0.54) {\t};}")
    a(r"\node[micro, anchor=west] at (0.4,-0.92) {solid = deterministic flow \quad dashed = hypothetical \quad dotted = derived/ascension};")
    a(r"\end{scope}")

    # ================= PANEL C : matcher pipeline =================
    a(r"\begin{scope}[shift={(0,6.3)}]")
    a(r"\node[pletter] at (-0.05,4.05) {C};")
    a(r"\node[ttl] at (9.0,4.05) {Inside the matcher: exact-anytime structure mapping (SME core)};")
    xs = [1.7, 5.0, 8.7, 12.4, 16.2]
    # helper mini-dag drawn inline per stage
    def mini(cx, cy, tag, col="h"):
        out = []
        out.append(rf"\node[{col}node] ({tag}t) at ({cx},{cy+0.6}) {{}};")
        out.append(rf"\node[fnode] ({tag}l) at ({cx-0.5},{cy-0.25}) {{}};")
        out.append(rf"\node[fnode] ({tag}r) at ({cx+0.5},{cy-0.25}) {{}};")
        out.append(rf"\draw[horel] ({tag}t) -- ({tag}l); \draw[horel] ({tag}t) -- ({tag}r);")
        return out
    cy = 2.3
    # 1 seeding (two dags + dotted correspondences)
    L += mini(xs[0]-0.75, cy, "s1b"); L += mini(xs[0]+0.75, cy, "s1t")
    for p in ("t", "l", "r"):
        a(rf"\draw[dotted, line width=0.5pt, color=black!45] (s1b{p}) -- (s1t{p});")
    # 2 kernels (shaded)
    a(rf"\begin{{scope}}[on background layer]\fill[smac!10, rounded corners=4pt] ({xs[1]-1.5},{cy-0.75}) rectangle ({xs[1]+1.5},{cy+1.0});\end{{scope}}")
    L += mini(xs[1]-0.75, cy, "s2b"); L += mini(xs[1]+0.75, cy, "s2t")
    for p in ("t", "l", "r"):
        a(rf"\draw[dotted, line width=0.5pt, color=black!45] (s2b{p}) -- (s2t{p});")
    # 3 conflict graph -> MWIS
    a(rf"\node[circle, draw=smac, fill=smac, text=white, minimum size=5mm, font=\sffamily\fontsize{{4}}{{4.6}}\selectfont] (k1) at ({xs[2]-0.6},{cy+0.8}) {{k1}};")
    a(rf"\node[circle, draw=black!35, fill=g1, text=black!60, minimum size=5mm, font=\sffamily\fontsize{{4}}{{4.6}}\selectfont] (k2) at ({xs[2]+0.6},{cy+0.8}) {{k2}};")
    a(rf"\node[circle, draw=black!35, fill=g1, text=black!60, minimum size=5mm, font=\sffamily\fontsize{{4}}{{4.6}}\selectfont] (k3) at ({xs[2]-0.6},{cy-0.55}) {{k3}};")
    a(rf"\node[circle, draw=smac, fill=smac, text=white, minimum size=5mm, font=\sffamily\fontsize{{4}}{{4.6}}\selectfont] (k4) at ({xs[2]+0.6},{cy-0.55}) {{k4}};")
    a(r"\draw[line width=0.8pt, color=inf] (k1) -- (k2); \draw[line width=0.8pt, color=inf] (k2) -- (k3);")
    a(r"\node[draw=ho, line width=0.8pt, ellipse, minimum width=7mm, minimum height=7mm, fit=(k1)] {};")
    a(r"\node[draw=ho, line width=0.8pt, ellipse, minimum width=7mm, minimum height=7mm, fit=(k4)] {};")
    # 4 trickle-down (bars on nodes)
    L += mini(xs[3], cy, "s4")
    a(rf"\fill[vio] ({xs[3]+0.16},{cy+0.52}) rectangle ++(0.55,0.16);")
    a(rf"\fill[vio] ({xs[3]-0.34},{cy-0.33}) rectangle ++(0.32,0.14);")
    a(rf"\fill[vio] ({xs[3]+0.66},{cy-0.33}) rectangle ++(0.32,0.14);")
    a(rf"\node[micro] at ({xs[3]},{cy+1.25}) {{$s(h)=\sigma_0\!\cdot\!\mathrm{{asc}}+\gamma\sum s(\mathrm{{parents}})$}};")
    # 5 candidate inference
    L += mini(xs[4]-0.55, cy, "s5b")
    a(rf"\node[fnode] (s5bc) at ({xs[4]-0.55},{cy-1.1}) {{}};")
    a(rf"\draw[forel] (s5bl) -- (s5bc);")
    L += mini(xs[4]+0.55, cy, "s5t")
    a(rf"\node[circle, draw=inf, dashed, fill=white, text=inf, minimum size=5mm, font=\sffamily\fontsize{{4.6}}{{5.2}}\selectfont] (s5q) at ({xs[4]+0.55},{cy-1.1}) {{?}};")
    a(rf"\draw[->, dashed, line width=0.7pt, color=inf] (s5bc) -- (s5q);")
    labels = ["1\\, seed match\\\\hypotheses", "2\\, close support\\\\$\\rightarrow$ kernels",
              "3\\, conflicts $\\rightarrow$ MWIS\\\\(CP-SAT, gap logged)",
              "4\\, systematicity\\\\scoring", "5\\, project inference\\\\(:hypothetical)"]
    for x, lbl in zip(xs, labels):
        a(rf"\node[sub] at ({x},0.45) {{{lbl}}};")
    for x0, x1 in zip(xs[:-1], xs[1:]):
        a(rf"\node[color=black!45, font=\Large] at ({(x0+x1)/2},{cy}) {{$\blacktriangleright$}};")
    a(r"\end{scope}")

    # ================= PANEL D : coverage matrix =================
    a(r"\begin{scope}[shift={(0,0.0)}]")
    a(r"\node[pletter] at (-0.05,5.55) {D};")
    a(r"\node[ttl] at (4.3,5.55) {Pre-registered battery: tasks $\times$ methods};")
    tasks = [("SSB (synthetic gold)", "queued", "SBD"),
             ("BGL$\\rightarrow$Spirit", "done", "ALL"),
             ("BGL$\\rightarrow$Thunderbird", "done", "ALL"),
             ("HDFS$\\rightarrow$OpenStack", "done", "ALL"),
             ("HDFS failure families", "running", "SBD"),
             ("BGL triage", "running", "ALL"),
             ("BugsInPy (code)", "queued", "SBD"),
             ("Liberty haystack", "queued", "SBH")]
    methods = ["SMA", "BM25", "Dense", "RRF", "Rerank", "KG", "Hippo"]
    cover = {"SBD": {"SMA", "BM25", "Dense"},
             "SBH": {"SMA", "BM25", "Dense", "RRF"}, "ALL": set(methods)}
    cx0, dx = 3.5, 0.62
    cy0, dy = 3.95, 0.45
    # ribbon groups sit ABOVE the rotated method labels (no collision)
    rib_lo, rib_hi = cy0 + 1.18, cy0 + 1.46
    groups = [("structural", [0], "ribA"), ("lexical", [1, 2], "ribB"),
              ("hybrid", [3, 4], "ribC"), ("graph", [5, 6], "ribD")]
    for name, cols, col in groups:
        xa = cx0 + (min(cols) - 0.45) * dx
        xb = cx0 + (max(cols) + 0.45) * dx
        a(rf"\fill[{col}, rounded corners=1.5pt] ({xa},{rib_lo}) rectangle ({xb},{rib_hi});")
        a(rf"\node[font=\sffamily\fontsize{{4}}{{4.8}}\selectfont, color=inkmid] at ({(xa+xb)/2},{(rib_lo+rib_hi)/2}) {{{name}}};")
        a(rf"\begin{{scope}}[on background layer]\fill[{col}!55] ({xa},{cy0-7*dy-0.05}) rectangle ({xb},{cy0+0.30});\end{{scope}}")
    for j, m in enumerate(methods):
        col = "smac" if m == "SMA" else "black!60"
        wt = r"\bfseries" if m == "SMA" else ""
        a(rf"\node[anchor=south west, rotate=40, font=\sffamily{wt}\fontsize{{4.2}}{{5}}\selectfont, color={col}] at ({cx0+j*dx-0.07},{cy0+0.36}) {{{m}}};")
    a(rf"\node[anchor=south west, rotate=40, font=\sffamily\fontsize{{4.2}}{{5}}\selectfont, color=black!55] at ({cx0+7.05*dx-0.07},{cy0+0.36}) {{status}};")
    for i, (task, status, cv) in enumerate(tasks):
        yy = cy0 - i * dy
        a(rf"\node[anchor=east, font=\sffamily\fontsize{{4.4}}{{5.2}}\selectfont, color=black!75] at ({cx0-0.55*dx},{yy}) {{{task}}};")
        for j, m in enumerate(methods):
            if m in cover[cv]:
                col = "smac" if m == "SMA" else "black!50"
                a(rf"\node[font=\sffamily\fontsize{{5}}{{5.5}}\selectfont, color={col}] at ({cx0+j*dx},{yy}) {{\checkmark}};")
        # status glyph
        sx = cx0 + 7.05 * dx
        if status == "done":
            a(rf"\fill[smac] ({sx},{yy}) circle (1.3mm);")
        elif status == "running":
            a(rf"\draw[kgc2, line width=0.5pt] ({sx},{yy}) circle (1.3mm); \fill[kgc2] ({sx},{yy}) -- ++(0,1.3mm) arc (90:270:1.3mm) -- cycle;")
        else:
            a(rf"\draw[black!45, line width=0.5pt] ({sx},{yy}) circle (1.3mm);")
    # status legend
    a(rf"\fill[smac] (0.5,0.55) circle (1.1mm); \node[micro, anchor=west] at (0.7,0.55) {{confirmatory complete}};")
    a(rf"\draw[kgc2, line width=0.5pt] (3.5,0.55) circle (1.1mm); \fill[kgc2] (3.5,0.55) -- ++(0,1.1mm) arc (90:270:1.1mm) -- cycle; \node[micro, anchor=west] at (3.7,0.55) {{running}};")
    a(rf"\draw[black!45, line width=0.5pt] (5.2,0.55) circle (1.1mm); \node[micro, anchor=west] at (5.4,0.55) {{queued}};")
    a(r"\node[micro, align=center] at (4.3,0.1) {seeds 201--205 / 41,43, frozen at prereg-v1 $\cdot$ deterministic extraction (\$0 LLM tokens)\\statistics: paired bootstrap + Holm--Bonferroni + Cliff's $\delta$};")
    a(r"\end{scope}")

    # ================= PANEL E : T1 transfer bars =================
    a(r"\begin{scope}[shift={(9.7,0.0)}]")
    a(r"\node[pletter] at (-0.05,5.55) {E};")
    a(r"\node[ttl, text width=8cm] at (4.4,5.55) {Confirmatory cross-system transfer (T1):\\label-hit@1, mean $\pm$ s.d., 5 seeds};")
    X0, XR = 0.95, 8.5
    ay0, ah = 0.95, 3.55
    def Y(v):
        return ay0 + v * ah
    a(rf"\draw[line width=0.5pt, color=black!55] ({X0},{ay0}) -- ({X0},{ay0+ah});")
    a(rf"\draw[line width=0.5pt, color=black!55] ({X0},{ay0}) -- ({XR},{ay0});")
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        a(rf"\draw[line width=0.4pt, color=black!55] ({X0-0.06},{Y(v)}) -- ({X0},{Y(v)});")
        a(rf"\node[micro, anchor=east] at ({X0-0.10},{Y(v)}) {{{v:.2f}}};")
    a(rf"\draw[dotted, line width=0.5pt, color=black!50] ({X0},{Y(0.5)}) -- ({XR},{Y(0.5)});")
    a(rf"\node[micro, anchor=west] at ({XR-0.45},{Y(0.5)+0.12}) {{chance}};")
    a(rf"\node[micro, rotate=90] at (0.08,{ay0+ah/2}) {{label-hit@1}};")
    group_w = 2.4
    starts = [X0 + 0.3, X0 + 2.9, X0 + 5.5]
    bw = group_w / len(BARMETH)
    for gi, (leg, lbl) in enumerate(LEGS):
        gx = starts[gi]
        for bi, m in enumerate(BARMETH):
            mean, sd = d[leg][m]
            x = gx + bi * bw
            a(rf"\fill[{BARCOL[m]}] ({x},{ay0}) rectangle ({x+bw*0.86},{Y(mean)});")
            if m == "SMA":
                a(rf"\draw[black, line width=0.4pt] ({x},{ay0}) rectangle ({x+bw*0.86},{Y(mean)});")
            if sd > 0:
                xc = x + bw * 0.43
                a(rf"\draw[line width=0.5pt, color=black!70] ({xc},{Y(max(mean-sd,0))}) -- ({xc},{Y(min(mean+sd,1))});")
                a(rf"\draw[line width=0.5pt, color=black!70] ({xc-0.05},{Y(min(mean+sd,1))}) -- ({xc+0.05},{Y(min(mean+sd,1))});")
                a(rf"\draw[line width=0.5pt, color=black!70] ({xc-0.05},{Y(max(mean-sd,0))}) -- ({xc+0.05},{Y(max(mean-sd,0))});")
        a(rf"\node[micro] at ({gx+group_w/2},{ay0-0.18}) {{{lbl}}};")
        best, delta, p = d[leg]["_best"]
        star = ("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s.")
        ann_y = (Y(0.88), Y(1.00), Y(0.63))[gi]
        a(rf"\node[font=\sffamily\fontsize{{4.2}}{{5}}\selectfont, color=black!70] at ({gx+group_w/2},{ann_y}) {{$\Delta$\,=\,{delta:+.2f} vs {best} ({star})}};")
    # method legend: horizontal row along the bottom (clear of plot + annotations)
    leg_items = [("SMA", "smac"), ("BM25", "g1"), ("Dense", "g2"), ("Hyb-RRF", "g3"),
                 ("Hyb+RR", "g4"), ("KG-PPR", "kgc"), ("HippoRAG", "kgc2")]
    lx = X0
    for name, col in leg_items:
        a(rf"\fill[{col}] ({lx},0.12) rectangle ++(0.15,0.15);")
        a(rf"\node[micro, anchor=west] at ({lx+0.18},0.195) {{{name}}};")
        lx += 1.08
    a(r"\end{scope}")

    return PREAMBLE + "\n".join(L) + "\n" + FOOT


OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(build(), encoding="utf-8")
print("wrote", OUT)
