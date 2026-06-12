"""Individual schematic figures as standalone TikZ (xelatex/Arial).

Emits 4 standalone .tex into paper/figures/individual/:
  fig01_architecture, fig02_representation, fig03_matcher, fig04_mechanism.
Same palette/styles as scripts/figure1_tikz.py. Compile via the tikz skill.
"""
from __future__ import annotations
import pathlib

OUT = pathlib.Path(__file__).resolve().parents[1] / "paper" / "figures" / "individual"
OUT.mkdir(parents=True, exist_ok=True)

PRE = r"""\documentclass[border=3pt]{standalone}
\usepackage{amsmath}\usepackage{amssymb}
\usepackage{fontspec}\setsansfont{Arial}\renewcommand{\familydefault}{\sfdefault}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,calc,fit,backgrounds,shapes.geometric}
\definecolor{titlegray}{HTML}{5F6B78}\definecolor{ent}{HTML}{7B8794}
\definecolor{fo}{HTML}{5AA9C4}\definecolor{ho}{HTML}{2E6B86}
\definecolor{inf}{HTML}{D98A3D}\definecolor{vio}{HTML}{9B86C4}
\definecolor{smac}{HTML}{2E8AA6}\definecolor{kgc}{HTML}{E7B15A}
\definecolor{fillteal}{HTML}{E3EEF2}\definecolor{fillamber}{HTML}{FBF0E2}
\definecolor{fillgrey}{HTML}{F1F3F5}\definecolor{fillviolet}{HTML}{EFEAF6}
\definecolor{inkmid}{HTML}{3D4751}\definecolor{distract}{HTML}{B3403A}
\begin{document}
\begin{tikzpicture}[
  font=\sffamily,>=Stealth,
  ttl/.style={font=\sffamily\fontsize{8}{9}\selectfont\bfseries, color=titlegray, align=center},
  sub/.style={font=\sffamily\fontsize{6}{7}\selectfont, color=black!55, align=center},
  micro/.style={font=\sffamily\fontsize{5.2}{6}\selectfont, color=black!50, align=center},
  enode/.style={circle, draw=ent, fill=ent, text=white, font=\sffamily\fontsize{5.2}{6}\selectfont, inner sep=0pt, minimum size=6.5mm, align=center},
  fnode/.style={circle, draw=fo, fill=fo, text=white, font=\sffamily\fontsize{5.2}{6}\selectfont, inner sep=0pt, minimum size=7mm, align=center},
  hnode/.style={circle, draw=ho, fill=ho, text=white, font=\sffamily\fontsize{5.2}{6}\selectfont, inner sep=0pt, minimum size=7.5mm, align=center},
  pbox/.style={rounded corners=2pt, draw, align=center, inner sep=3pt, font=\sffamily\fontsize{6}{7}\selectfont},
  flowarr/.style={->, line width=0.7pt, color=black!55},
  horel/.style={->, line width=0.7pt, color=ho},
  forel/.style={->, line width=0.6pt, color=fo!80},
]
"""
FOOT = "\n\\end{tikzpicture}\n\\end{document}\n"


def write(name, body):
    (OUT / f"{name}.tex").write_text(PRE + body + FOOT, encoding="utf-8")
    print(f"  {name}.tex")


# =========================================================== F1 architecture =
ARCH = r"""
\node[ttl] at (7,5.5) {Deterministic write path, certified read path, one gated LLM};
% write lane
\node[micro,rotate=90,color=black!45] at (-0.3,4.4) {WRITE};
\node[pbox,draw=black!40,fill=white,text width=2.0cm,minimum height=1.1cm] (art) at (1.2,4.4) {artifacts\\logs $\cdot$ code\\traces $\cdot$ obs};
\node[pbox,draw=fo,fill=fillteal,text width=2.3cm,minimum height=1.1cm] (enc) at (4.3,4.4) {encoders\\rules; versioned;\\byte-deterministic};
\node[cylinder,shape border rotate=90,draw=smac,fill=fillteal,aspect=0.28,minimum width=2.5cm,minimum height=1.5cm,inner sep=2pt,font=\sffamily\fontsize{5.6}{6.4}\selectfont,align=center] (store) at (7.6,4.45) {case store\\BLAKE3 ids\\+ WAL};
\node[pbox,draw=fo,fill=fillteal,text width=2.3cm,minimum height=1.1cm] (idx) at (10.9,4.4) {index\\functor postings\\+ WL-1 vectors};
\draw[flowarr] (art)--(enc); \draw[flowarr] (enc)--(store); \draw[flowarr] (store)--(idx);
% read lane
\node[micro,rotate=90,color=black!45] at (-0.3,1.7) {READ};
\node[pbox,draw=black!40,fill=white,text width=1.6cm,minimum height=1.1cm] (q) at (1.1,1.7) {query\\artifact};
\node[pbox,draw=fo,fill=fillteal,text width=2.4cm,minimum height=1.1cm] (mac) at (4.0,1.7) {\textbf{MAC} shortlist\\admissible bound\\orders candidates};
\node[pbox,draw=ho,fill=ho!15,text width=2.4cm,minimum height=1.1cm] (fac) at (7.3,1.7) {\textbf{FAC}: SME align\\best-first,\\certified top-$k$};
\node[pbox,draw=inf,fill=white,text width=2.0cm,minimum height=1.1cm] (rec) at (10.3,1.7) {receipts:\\maps $\cdot$ scores\\$\cdot$ inferences};
\draw[flowarr] (q)--(mac); \draw[flowarr] (mac)--(fac); \draw[flowarr] (fac)--(rec);
\draw[flowarr,dashed] (store)-- node[micro,right=1pt]{single source of truth} (7.6,2.6);
\node[pbox,draw=inf,fill=fillamber,text width=2.3cm,minimum height=1.2cm] (llm) at (12.8,0.1) {\textbf{LLM} verbalizes\\receipts only:\\CITE or ABSTAIN};
\node[draw=black!35,dashed,rounded corners=3pt,fit=(llm),inner sep=4pt,label={[micro]above:the only LLM --- never writes facts}] {};
\draw[flowarr] (rec)|-(llm);
% legend
\foreach \x/\c/\t in {0.4/ent/entity,3.0/fo/{first-order relation},6.8/ho/{higher-order relation},10.2/inf/{candidate inference},13.0/vio/{lattice ascension}}
 {\fill[\c] (\x,-1.1) rectangle ++(0.28,0.28); \node[micro,anchor=west] at (\x+0.34,-0.96) {\t};}
\node[micro,anchor=west] at (0.4,-1.6) {solid = deterministic flow \quad dashed = hypothetical / derived};
"""
write("fig01_architecture", ARCH)

# ========================================================== F2 representation =
REP = r"""
\node[ttl] at (5.5,5.6) {Representation: artifact $\rightarrow$ typed predicate DAG};
\node[draw=black!40,fill=fillgrey,rounded corners=3pt,align=left,inner sep=4pt,font=\ttfamily\fontsize{5}{6}\selectfont,text width=5.6cm] (log) at (2.9,4.7) {1117841440 R63-M0 ciod: timeout tree link\\1117841452 R63-M0 ciod: retry tree link\\1117841498 R63-M0 kernel panic};
\node[micro] at (2.9,5.3) {raw session (BGL)};
\node[sub] (enc) at (2.9,3.7) {Tier-0 encoder (deterministic rules)};
\draw[flowarr] (2.9,4.25)--(2.9,3.95); \draw[flowarr] (2.9,3.45)--(2.9,3.1);
% order axis
\draw[->,line width=0.6pt,color=black!45] (0.3,0.5) -- (0.3,2.95);
\node[micro,rotate=90] at (0.0,1.7) {order};
\foreach \o/\y in {0/0.7,1/1.7,2/2.7} {\node[micro] at (0.55,\y) {\o};}
\node[enode] (e1) at (2.1,0.7) {R63-M0}; \node[enode] (e2) at (4.6,0.7) {treeLink};
\node[fnode] (f1) at (1.7,1.7) {timeout}; \node[fnode] (f2) at (3.4,1.7) {retry}; \node[fnode] (f3) at (5.2,1.7) {failure};
\node[hnode] (h1) at (2.55,2.7) {cause}; \node[hnode] (h2) at (4.3,2.7) {cause};
\foreach \h/\a/\i in {h1/f1/1,h1/f2/2,h2/f2/1,h2/f3/2} {\draw[horel] (\h)--(\a) node[midway,font=\sffamily\fontsize{4.4}{5}\selectfont,color=black!45,inner sep=1pt]{\i};}
\foreach \f/\e in {f1/e1,f1/e2,f2/e1,f2/e2,f3/e1} {\draw[forel] (\f)--(\e);}
\node[micro,align=left,text width=3.6cm] at (8.3,2.0) {\textit{shared sub-expressions are hash-consed $\Rightarrow$ the case is a DAG, not a bag; this is what makes systematicity a graph property}};
\node[pbox,draw=smac,fill=fillteal,text=black,font=\sffamily\fontsize{5.2}{6.2}\selectfont,text width=10.6cm] at (5.5,-0.15) {\texttt{(cause (timeout R63-M0 treeLink) (retry R63-M0 treeLink))} \quad case\_id = BLAKE3(canonical) = 6c1f6a\ldots};
"""
write("fig02_representation", REP)

# ============================================================== F3 matcher ===
def mini(cx, cy, tag, s=0.45):
    return (rf"\node[hnode] ({tag}t) at ({cx},{cy+s*1.6}) {{}};"
            rf"\node[fnode] ({tag}l) at ({cx-s},{cy-s*0.8}) {{}};"
            rf"\node[fnode] ({tag}r) at ({cx+s},{cy-s*0.8}) {{}};"
            rf"\draw[horel] ({tag}t)--({tag}l); \draw[horel] ({tag}t)--({tag}r);")


MATCH = r"\node[ttl] at (9,5.6) {Inside the matcher: exact-anytime structure mapping (SME core)};" + "\n"
xs = [1.6, 5.2, 9.0, 12.8, 16.4]
cy = 3.0
# 1 seeding
MATCH += mini(xs[0] - 0.7, cy, "s1b") + mini(xs[0] + 0.7, cy, "s1t")
for p in ("t", "l", "r"):
    MATCH += rf"\draw[dotted,line width=0.7pt,color=black!45] (s1b{p})--(s1t{p});"
# 2 kernels
MATCH += rf"\begin{{scope}}[on background layer]\fill[smac!10,rounded corners=5pt] ({xs[1]-1.5},{cy-1.0}) rectangle ({xs[1]+1.5},{cy+1.4});\end{{scope}}"
MATCH += mini(xs[1] - 0.7, cy, "s2b") + mini(xs[1] + 0.7, cy, "s2t")
for p in ("t", "l", "r"):
    MATCH += rf"\draw[dotted,line width=0.7pt,color=black!45] (s2b{p})--(s2t{p});"
# 3 conflict graph -> MWIS
MATCH += (rf"\node[circle,draw=smac,fill=smac,text=white,minimum size=7mm,font=\sffamily\fontsize{{5}}{{6}}\selectfont] (k1) at ({xs[2]-0.75},{cy+0.85}) {{k1}};"
          rf"\node[circle,draw=black!35,fill=fillgrey,text=black!60,minimum size=7mm,font=\sffamily\fontsize{{5}}{{6}}\selectfont] (k2) at ({xs[2]+0.75},{cy+0.85}) {{k2}};"
          rf"\node[circle,draw=black!35,fill=fillgrey,text=black!60,minimum size=7mm,font=\sffamily\fontsize{{5}}{{6}}\selectfont] (k3) at ({xs[2]-0.75},{cy-0.7}) {{k3}};"
          rf"\node[circle,draw=smac,fill=smac,text=white,minimum size=7mm,font=\sffamily\fontsize{{5}}{{6}}\selectfont] (k4) at ({xs[2]+0.75},{cy-0.7}) {{k4}};"
          r"\draw[line width=1pt,color=distract] (k1)--(k2); \draw[line width=1pt,color=distract] (k2)--(k3);"
          r"\node[draw=ho,line width=1pt,circle,minimum size=10mm] at (k1) {}; \node[draw=ho,line width=1pt,circle,minimum size=10mm] at (k4) {};"
          rf"\node[micro] at ({xs[2]},{cy-1.7}) {{max-weight independent set\\(CP-SAT, gap logged)}};")
# 4 trickle-down
MATCH += mini(xs[3], cy, "s4")
MATCH += (rf"\fill[vio] ({xs[3]+0.22},{cy+0.62}) rectangle ++(0.7,0.22);"
          rf"\fill[vio] ({xs[3]-0.62},{cy-0.46}) rectangle ++(0.42,0.18);"
          rf"\fill[vio] ({xs[3]+0.7},{cy-0.46}) rectangle ++(0.42,0.18);"
          rf"\node[micro] at ({xs[3]},{cy+1.65}) {{$s(h)=\sigma_0\cdot\mathrm{{asc}}+\gamma\sum s(\mathrm{{parents}})$}};")
# 5 candidate inference
MATCH += mini(xs[4] - 0.7, cy, "s5b")
MATCH += rf"\node[fnode] (s5bc) at ({xs[4]-0.7},{cy-1.7}) {{}}; \draw[forel] (s5bl)--(s5bc);"
MATCH += mini(xs[4] + 0.7, cy, "s5t")
MATCH += (rf"\node[circle,draw=distract,dashed,fill=white,text=distract,minimum size=7mm,font=\sffamily\fontsize{{6}}{{7}}\selectfont] (s5q) at ({xs[4]+0.7},{cy-1.7}) {{?}};"
          rf"\draw[->,dashed,line width=0.9pt,color=distract] (s5bc)--(s5q);")
labels = ["1\\, seed match\\\\hypotheses", "2\\, close support\\\\$\\rightarrow$ kernels",
          "3\\, conflicts $\\rightarrow$ MWIS\\\\(exact, gap logged)", "4\\, systematicity\\\\scoring",
          "5\\, project inference\\\\(:hypothetical)"]
for x, lbl in zip(xs, labels):
    MATCH += rf"\node[sub] at ({x},0.4) {{{lbl}}};"
for x0, x1 in zip(xs[:-1], xs[1:]):
    MATCH += rf"\node[color=black!45,font=\Large] at ({(x0+x1)/2},{cy}) {{$\blacktriangleright$}};"
write("fig03_matcher", MATCH)

# ============================================================ F4 mechanism ===
# NeuroPath-style cross-system alignment with candidate inference.
MECH = r"""
\node[ttl] at (8,7.0) {Cross-system structure mapping: zero shared vocabulary, bridged by the predicate lattice};
\node[sub,color=ho] at (3,6.2) {Base: stored BGL incident};
\node[sub,color=smac] at (13,6.2) {Target: new Spirit session (query)};
% base DAG (left)
\node[hnode] (bh1) at (3,5.0) {cause};
\node[fnode,text width=1.6cm,inner sep=1pt,minimum size=0pt] (bt) at (1.7,3.4) {\fontsize{4.6}{5.4}\selectfont timeout\\R63-M0,treeLink};
\node[fnode,text width=1.5cm,inner sep=1pt,minimum size=0pt] (br) at (4.3,3.4) {\fontsize{4.6}{5.4}\selectfont retry\\R63-M0,treeLink};
\node[hnode] (bh2) at (4.3,1.9) {cause};
\node[fnode,text width=1.4cm,inner sep=1pt,minimum size=0pt] (bf) at (4.3,0.4) {\fontsize{4.6}{5.4}\selectfont failure\\R63-M0};
\draw[horel] (bh1)--(bt); \draw[horel] (bh1)--(br); \draw[horel] (bh2)--(br); \draw[horel] (bh2)--(bf);
% target DAG (right) - disjoint vocabulary
\node[hnode,draw=smac,fill=smac] (th1) at (13,5.0) {q\_cause};
\node[fnode,fill=fo,text width=1.5cm,inner sep=1pt,minimum size=0pt] (tt) at (11.7,3.4) {\fontsize{4.6}{5.4}\selectfont q\_timeout\\sn-a12,fabric};
\node[fnode,fill=fo,text width=1.5cm,inner sep=1pt,minimum size=0pt] (tr) at (14.3,3.4) {\fontsize{4.6}{5.4}\selectfont q\_retry\\sn-a12,fabric};
\node[circle,draw=distract,dashed,fill=white,text=distract,text width=1.3cm,inner sep=1pt,minimum size=10mm,font=\sffamily\fontsize{4.6}{5.4}\selectfont,align=center] (tf) at (14.3,0.4) {q\_failure\\sn-a12};
\draw[horel,color=smac] (th1)--(tt); \draw[horel,color=smac] (th1)--(tr);
% candidate inference projection
\draw[->,dashed,line width=1pt,color=distract] (tr)--(tf);
\node[micro,color=distract,align=left,text width=3.2cm] at (15.6,0.4) {\textbf{candidate inference}\\(status: hypothetical;\\verify or abstain) ---\\an object RAG cannot emit};
% correspondences (bundled)
\draw[<->,dotted,line width=0.8pt,color=black!45] (bh1) to[bend left=12] (th1);
\draw[<->,dotted,line width=0.8pt,color=black!45] (bt) to[bend left=8] (tt);
\draw[<->,dotted,line width=0.8pt,color=black!45] (br) to[bend right=8] (tr);
\node[sub,align=center] at (8,5.4) {match hypotheses\\(injective, parallel-connected)};
% lattice ascension inset
\node[draw=vio,fill=fillviolet,rounded corners=2pt,align=center,inner sep=3pt,font=\sffamily\fontsize{4.8}{5.6}\selectfont] (lat) at (8,2.7) {predicate lattice L\\ timeout $\sqsubseteq$ \textbf{timeoutEvent} $\sqsupseteq$ q\_timeout\\ ascend $\delta{\le}2$ at penalty $\rho^{\mathrm{dist}}$};
\draw[->,dotted,line width=0.8pt,color=vio] (bt) to[bend right=10] (lat);
\draw[->,dotted,line width=0.8pt,color=vio] (tt) to[bend left=10] (lat);
% bottom contrast strip
\node[sub,align=left,text width=15cm] at (8,-1.0) {\textit{Lexical (BM25/dense) rank the surface distractor (same words, broken structure); SMA ranks the structural analog and projects the missing consequence. Entity map: R63-M0 $\leftrightarrow$ sn-a12, treeLink $\leftrightarrow$ fabric --- zero shared names.}};
"""
write("fig04_mechanism", MECH)

print("schematic tex written")
