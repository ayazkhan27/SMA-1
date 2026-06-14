#!/usr/bin/env python3
"""Migrate the SMA-1 manuscript onto the official Springer Nature sn-jnl
(sn-nature) template, preserving every number/citation/equation byte-for-byte.
Only structural commands are transformed."""
import re, pathlib

HERE = pathlib.Path(__file__).parent
backup = (HERE / "sma_nature_mi_xelatex_backup.tex").read_text()
supp = (HERE / "supplement.tex").read_text()

# ---- 1. extract the abstract body (strip leading \noindent) ----
abs = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", backup, re.DOTALL).group(1)
abs_body = abs.replace("\\noindent", "").strip()

# ---- 2. extract the body: from \msec{Main} up to \printbibliography ----
body = backup[backup.index("\\msec{Main}"):backup.index("\\printbibliography")]

# split off the Data-availability tail (becomes back matter, keep its run-ins)
DA = "\\msec{Data and code availability}"
main_body, da_body = body.split(DA)

# ---- 3. transform MAIN BODY ----
# 3a. section heads
main_body = main_body.replace("\\msec{Main}", "\\section{Introduction}\\label{sec:intro}")
main_body = main_body.replace("\\msec{Results}", "\\section{Results}\\label{sec:results}")
main_body = main_body.replace("\\msec{Discussion}", "\\section{Discussion}\\label{sec:discussion}")
main_body = main_body.replace("\\msec{Methods}", "\\section{Methods}\\label{sec:methods}")

# 3b. paragraph-leading \textit{...} run-ins -> \subsection{...}
def runin_to_subsec(m):
    title = re.sub(r"\s+", " ", m.group(1)).strip().rstrip(".")
    return "\\subsection{%s}\n" % title
main_body = re.sub(r"(?m)^\\textit\{([^}]+)\}", runin_to_subsec, main_body)

# 3c. swap conceptual figures to the dense TikZ versions
main_body = main_body.replace("design/fig1_overview_crop.pdf", "tikz/fig1_overview.pdf")
main_body = main_body.replace("design/figM_architecture_crop.pdf", "tikz/figM_architecture.pdf")
# figure labels (after the includegraphics line they follow)
main_body = main_body.replace(
    "\\includegraphics[width=\\linewidth]{tikz/fig1_overview.pdf}",
    "\\includegraphics[width=\\textwidth]{tikz/fig1_overview.pdf}")
main_body = main_body.replace(
    "\\includegraphics[width=\\linewidth]{tikz/figM_architecture.pdf}",
    "\\includegraphics[width=\\textwidth]{tikz/figM_architecture.pdf}")
main_body = main_body.replace(
    "\\includegraphics[width=\\linewidth]{svg/figure2_results.pdf}",
    "\\includegraphics[width=\\textwidth]{svg/figure2_results.pdf}")
main_body = main_body.replace(
    "\\includegraphics[width=\\linewidth]{svg/figure5_trustworthy.pdf}",
    "\\includegraphics[width=\\textwidth]{svg/figure5_trustworthy.pdf}")

# 3d. fix the long unbreakable monospace loader token (158pt overfull)
main_body = main_body.replace(
    "\\texttt{load\\_obo/load\\_owl/load\\_owl\\_dir/load\\_rdflib/load\\_attack\\_stix/}\n"
    "\\texttt{load\\_cpc/load\\_mitre\\_xml}",
    "\\texttt{load\\_obo}, \\texttt{load\\_owl}, \\texttt{load\\_owl\\_dir}, "
    "\\texttt{load\\_rdflib}, \\texttt{load\\_attack\\_stix}, \\texttt{load\\_cpc} and "
    "\\texttt{load\\_mitre\\_xml}")

# 3e. grey the SMA (headline) rows + Total row in the two main tables
main_body = main_body.replace(
    "\\textbf{SMA (ours)} & \\textbf{0.949}",
    "\\rowcolor{smarow}\\textbf{SMA (ours)} & \\textbf{0.949}")
main_body = main_body.replace(
    "SMA & \\textbf{0.017}/0.18",
    "\\rowcolor{smarow}SMA & \\textbf{0.017}/0.18")
main_body = main_body.replace(
    "\\multicolumn{2}{l}{\\textbf{Total}}",
    "\\rowcolor{smarow}\\multicolumn{2}{l}{\\textbf{Total}}")

# ---- 4. transform DATA-AVAILABILITY tail (back matter; keep run-ins) ----
da_body = da_body.replace("https://github.com/ayazkhan27/sma-1",
                          "https://github.com/ayazkhan27/SMA-1")

# ---- 5. transform SUPPLEMENT -> appendix fragment ----
supp = supp[supp.index("\\clearpage"):] if "\\clearpage" in supp else supp
supp = supp.replace("\\clearpage\n", "").replace("\\clearpage", "")
# Unnumbered ED captions: the class \caption* prints a stray "Table N *" label.
# Swap to our \edcaption (argument braces already balanced in the source).
supp = supp.replace("\\caption*{", "\\edcaption{")
# Shrink only the over-wide ED tables to the text block; adjustbox max-width is a
# no-op for tables that already fit, so wrapping every tabular is safe.
supp = supp.replace("\\begin{tabular}", "\\begin{adjustbox}{max width=\\textwidth}\\begin{tabular}")
supp = supp.replace("\\end{tabular}", "\\end{tabular}\\end{adjustbox}")

# ---- 6. assemble ----
header = r"""\documentclass[pdflatex,sn-nature]{sn-jnl}

% Standard packages expected by the template (mirrors sn-article.tex);
% xcolor[table] additionally enables greyed table rows (\rowcolor).
\usepackage{graphicx}
\usepackage{multirow}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{booktabs}
\usepackage[table]{xcolor}
\usepackage{url}
\usepackage{adjustbox} % shrink any over-wide Extended Data table to the text block

\graphicspath{{./}{../figures/}{../figures/tikz/}{../figures/svg/}}
\definecolor{smarow}{rgb}{0.91,0.94,0.96} % light teal-grey for the SMA (headline) row

% Unnumbered Extended Data caption (the class \caption* prints a stray label);
% render the legend as a small justified paragraph instead.
\newcommand{\edcaption}[1]{\par{\small #1}\par\smallskip}

\begin{document}

\title[Align, Don't Retrieve]{Align, Don't Retrieve: Structure-Mapping Memory Grounds Language Models in Curated Ontologies}

\author*[1]{\fnm{Ayaz} \sur{Khan}}\email{aak2259@columbia.edu}

\affil*[1]{\orgname{Columbia University}, \orgaddress{\city{New York}, \state{NY}, \country{USA}}}

\abstract{__ABSTRACT__}

\keywords{structure mapping, analogical retrieval, ontology, retrieval-augmented generation, abstention, novelty detection}

\maketitle

"""
header = header.replace("__ABSTRACT__", abs_body)

footer = r"""
\backmatter

\bmhead{Data and code availability}
__DA__

\bmhead{Competing interests}
The author declares no competing interests.

\bibliography{references}

\begin{appendices}

__SUPP__

\end{appendices}

\end{document}
"""
footer = footer.replace("__DA__", da_body.strip()).replace("__SUPP__", supp.strip())

out = header + main_body.strip() + "\n" + footer
(HERE / "sma_nature_mi.tex").write_text(out)
print("wrote sma_nature_mi.tex (%d bytes)" % len(out))
print("subsections:", len(re.findall(r"\\subsection\{", out)))
print("sections:", re.findall(r"\\section\{([^}]+)\}", out))
print("rowcolors:", len(re.findall(r"\\rowcolor", out)))
print("resizeboxes:", len(re.findall(r"\\resizebox", out)))
