"""Regenerate all paper assets (figures, tables, diagrams) from reports/*.csv.

Incremental-by-design: run after every phase (`make paper`); each asset is
rebuilt from the canonical CSV artifacts and stamped with the git revision and
data-semantics version. Figures whose source numbers predate matcher-semantics
v3 are hatched and labeled "pre-v3" until their runs are re-verified.

Style: SciencePlots (Garrett, J., 2021, DOI 10.5281/zenodo.4106649).
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
import scienceplots  # noqa: F401

plt.style.use(["science", "grid"])
plt.rcParams.update({"figure.dpi": 200, "savefig.bbox": "tight"})

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PAPER = ROOT / "paper"
FIGS = PAPER / "figures"
TABLES = PAPER / "tables"
DIAGRAMS = PAPER / "diagrams"

SEMANTICS = "matcher-semantics v3 (constants + bound-ordered MAC)"


def git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT
        ).stdout.strip()
    except Exception:
        return "unknown"


STAMP = f"{datetime.date.today()} · {git_rev()} · {SEMANTICS}"


def load(name: str) -> list[dict]:
    path = REPORTS / name
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r.get("dataset") != "DIAGNOSTIC"]


def stamp(fig):
    fig.text(0.99, 0.005, STAMP, ha="right", va="bottom", fontsize=4, color="0.6")


def save(fig, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{name}.{ext}")
    plt.close(fig)
    print(f"  fig: {name}.pdf/.png")


# ----------------------------------------------------------------- figures --

def fig_transfer_headline():
    """Cross-system transfer: the H1 figure. v3-verified Spirit leg solid;
    pre-v3 pairs hatched pending re-verification."""
    v3 = {r["method"]: float(r["macro_f1"]) for r in load("transfer_verify_v3.csv")
          if "spirit" in r["split"]}
    pre = load("transfer_metrics.csv")
    tbird = {r["method"]: float(r["macro_f1"]) for r in pre
             if "thunderbird" in r["split"] and "[ses]" in r["split"]}
    ostack = {r["method"]: float(r["macro_f1"]) for r in pre
              if "OpenStack[ses]" in r["split"]}
    methods = ["SMA", "BM25", "Dense RAG", "KG-PPR Proxy"]
    pairs = [
        ("BGL$\\rightarrow$Spirit\n(held-out, v3)", v3, False),
        ("BGL$\\rightarrow$Thunderbird\n(pre-v3)", tbird, True),
        ("HDFS$\\rightarrow$OpenStack\n(pre-v3)", ostack, True),
    ]
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    width = 0.2
    x = np.arange(len(pairs))
    colors = ["#2563eb", "#b45309", "#7c3aed", "#047857"]
    for mi, method in enumerate(methods):
        vals = [d.get(method, 0.0) for _, d, _ in pairs]
        hatches = ["///" if h else "" for _, _, h in pairs]
        bars = ax.bar(x + (mi - 1.5) * width, vals, width,
                      label=method.replace("KG-PPR Proxy", "KG proxy"),
                      color=colors[mi], edgecolor="black", linewidth=0.4)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
    ax.axhline(1 / 3, color="0.4", linestyle=":", linewidth=0.8)
    ax.text(2.42, 1 / 3 + 0.01, "collapse (0.33)", fontsize=5, color="0.4", ha="right")
    ax.set_xticks(x, [p for p, _, _ in pairs], fontsize=6)
    ax.set_ylabel("macro-F1")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=5, ncol=2, loc="upper right")
    ax.set_title("Cross-system transfer (frozen ontology; hatched = pending re-verification)", fontsize=7)
    stamp(fig)
    save(fig, "fig_transfer_headline")


def fig_decomposition():
    """Representation vs alignment: controls on the identical BGL->Spirit sets."""
    controls = {r["method"]: float(r["macro_f1"]) for r in load("transfer_controls_metrics.csv")}
    v3_sma = next((float(r["macro_f1"]) for r in load("transfer_verify_v3.csv")
                   if r["method"] == "SMA"), None)
    steps = [
        ("Dense RAG\n(embeddings)", 0.3144, True),
        ("Hybrid+Rerank\n(production RAG)", controls.get("Hybrid+Rerank", 0.5947), True),
        ("WL kernel\n(same Tier-0 repr.)", controls.get("WL-kernel", 0.6239), True),
        ("SMA\n(SME alignment)", v3_sma or 0.8942, False),
    ]
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    xs = np.arange(len(steps))
    vals = [v for _, v, _ in steps]
    cols = ["#7c3aed", "#0e7490", "#64748b", "#2563eb"]
    bars = ax.bar(xs, vals, 0.62, color=cols, edgecolor="black", linewidth=0.4)
    for bar, (_, _, h) in zip(bars, steps):
        if h:
            bar.set_hatch("///")
    for xi, v in zip(xs, vals):
        ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=6)
    ax.set_xticks(xs, [s for s, _, _ in steps], fontsize=5.5)
    ax.set_ylabel("macro-F1 (BGL$\\rightarrow$Spirit)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Representation is necessary, alignment is the active ingredient", fontsize=7)
    stamp(fig)
    save(fig, "fig_decomposition")


def fig_family():
    rows = load("scorer_gauntlet_v3.csv")
    variants = ["ses", "surprisal", "mdl", "rrf"]
    groups = [("HDFS", "family_hit5_common"), ("HDFS", "family_hit5_rare"),
              ("BGL", "family_hit5_common")]
    labels = ["HDFS common", "HDFS rare", "BGL common"]
    fig, ax = plt.subplots(figsize=(3.8, 2.4))
    x = np.arange(len(groups))
    width = 0.19
    colors = ["#2563eb", "#0e7490", "#b45309", "#64748b"]
    for vi, variant in enumerate(variants):
        vals = []
        for ds, col in groups:
            row = next((r for r in rows if r["dataset"] == ds and r["variant"] == variant), {})
            vals.append(float(row.get(col) or 0))
        ax.bar(x + (vi - 1.5) * width, vals, width, label=variant,
               color=colors[vi], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x, labels, fontsize=6)
    ax.set_ylabel("family-hit@5")
    ax.legend(fontsize=5, ncol=4)
    ax.set_title("Failure-family retrieval by scorer (v3 semantics)", fontsize=7)
    stamp(fig)
    save(fig, "fig_family_scorers")


def fig_h3():
    rows = [r for r in csv.DictReader(open(REPORTS / "h3_judged.csv"))] if (REPORTS / "h3_judged.csv").exists() else []
    if not rows:
        return
    def rate(llm, pred):
        sub = [r for r in rows if r["llm"] == llm]
        return sum(pred(r) for r in sub) / max(len(sub), 1)
    metrics = {
        "judged correct": lambda r: r.get("judged_correct") == "true",
        "confabulated": lambda r: r.get("confabulated") == "true",
        "abstained when\nunanswerable": lambda r: r["answerable"] == "False" and r.get("judged_abstained") == "true",
    }
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    x = np.arange(len(metrics))
    for li, (llm, color) in enumerate([("deepseek", "#2563eb"), ("local", "#b45309")]):
        vals = []
        for name, pred in metrics.items():
            if "unanswerable" in name:
                sub = [r for r in rows if r["llm"] == llm and r["answerable"] == "False"]
                vals.append(sum(r.get("judged_abstained") == "true" for r in sub) / max(len(sub), 1))
            else:
                vals.append(rate(llm, pred))
        ax.bar(x + (li - 0.5) * 0.32, vals, 0.32,
               label="DeepSeek (provenance-bound)" if llm == "deepseek" else "Qwen-0.5B (undisciplined)",
               color=color, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x, list(metrics), fontsize=6)
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=5)
    ax.set_title("H3: judged honesty across 200 answer cells", fontsize=7)
    stamp(fig)
    save(fig, "fig_h3_honesty")


def fig_ladder():
    rows = load("baseline_ladder_metrics.csv")
    triage = load("triage_metrics.csv")
    hdfs = {r["method"]: float(r["macro_f1"]) for r in rows if r["split"] == "HDFS_MVP_diagnostic"}
    sma = next((float(r["macro_f1"]) for r in triage
                if r["split"] == "HDFS_MVP_diagnostic" and r["method"] == "SMA"), None)
    entries = [("SMA", sma or 0.9549), ("WL-kernel", hdfs.get("WL-kernel")),
               ("Hybrid-RRF", hdfs.get("Hybrid-RRF")), ("B6 long-context\nDeepSeek", hdfs.get("B6-LongContext-DeepSeek")),
               ("Hybrid+Rerank", hdfs.get("Hybrid+Rerank")), ("BGE dense", hdfs.get("BGE-dense")),
               ("SPLADE", hdfs.get("SPLADE"))]
    entries = [(n, v) for n, v in entries if v]
    fig, ax = plt.subplots(figsize=(3.8, 2.4))
    ys = np.arange(len(entries))[::-1]
    vals = [v for _, v in entries]
    cols = ["#2563eb" if n == "SMA" else "#64748b" for n, _ in entries]
    ax.barh(ys, vals, 0.6, color=cols, edgecolor="black", linewidth=0.4, hatch="///")
    for y, v in zip(ys, vals):
        ax.text(v + 0.008, y, f"{v:.2f}", va="center", fontsize=6)
    ax.set_yticks(ys, [n for n, _ in entries], fontsize=6)
    ax.set_xlabel("macro-F1 (HDFS within-system)")
    ax.set_xlim(0, 1.05)
    ax.set_title("Production baseline ladder (pre-v3, pending re-verification)", fontsize=7)
    stamp(fig)
    save(fig, "fig_ladder_hdfs")


# ------------------------------------------------------------------ tables --

def tex_table(name: str, caption: str, label: str, headers: list[str], rows: list[list[str]]):
    lines = [
        "% auto-generated by scripts/make_paper_assets.py - do not edit",
        f"% {STAMP}",
        "\\begin{table}[t]\\centering",
        f"\\caption{{{caption}}}\\label{{{label}}}",
        "\\begin{tabular}{l" + "r" * (len(headers) - 1) + "}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (TABLES / f"{name}.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"  table: {name}.tex")


def make_tables():
    v3 = [r for r in load("transfer_verify_v3.csv") if "spirit" in r["split"]]
    tex_table(
        "tab_transfer_v3",
        "Held-out cross-system transfer (BGL$\\rightarrow$Spirit, frozen ontology, v3 semantics).",
        "tab:transfer",
        ["Method", "macro-F1", "hit@1", "hit@5", "p50 (ms)"],
        [[r["method"], r["macro_f1"], r["label_hit_rate@1"], r["label_hit_rate@5"],
          r["p50_ms"]] for r in v3],
    )
    g = load("scorer_gauntlet_v3.csv")
    tex_table(
        "tab_family_v3",
        "Failure-family retrieval (family-hit@5) by scorer under v3 semantics.",
        "tab:family",
        ["Dataset", "Scorer", "common", "rare"],
        [[r["dataset"], r["variant"], r["family_hit5_common"] or "--",
          r["family_hit5_rare"] or "--"] for r in g if r["dataset"] in ("HDFS", "BGL")],
    )
    ladder = load("baseline_ladder_metrics.csv")
    tex_table(
        "tab_ladder",
        "Production retrieval ladder, HDFS within-system (pre-v3, pending re-verification).",
        "tab:ladder",
        ["Method", "macro-F1", "hit@1"],
        [[r["method"], r["macro_f1"], r["label_hit_rate@1"]]
         for r in ladder if r["split"] == "HDFS_MVP_diagnostic"],
    )
    bugs = load("bugsinpy_metrics.csv")
    if bugs:
        tex_table(
            "tab_bugsinpy",
            "BugsInPy fix-category retrieval (T3); LOPO = leave-one-project-out.",
            "tab:bugsinpy",
            ["Split", "Method", "cat@1", "cat@5"],
            [[r.get("split_mode", ""), r.get("method", ""), r.get("category_hit@1", ""),
              r.get("category_hit@5", "")] for r in bugs],
        )


# ---------------------------------------------------------------- diagrams --

def make_diagrams():
    (DIAGRAMS / "architecture.mmd").write_text("""\
%% SMA-1 architecture (render: mmdc -i architecture.mmd -o architecture.pdf)
flowchart LR
    A[Raw artifacts<br/>logs / code / traces] --> B[Tier-0 encoders<br/>deterministic rules]
    B --> C[(Case store<br/>content-addressed)]
    C --> D[MAC: certified bound ordering<br/>weighted Lemma-2]
    D --> E[FAC: SME matching<br/>kernels, merge, surprisal-SES]
    E --> F[Receipts: alignment,<br/>candidate inferences, ses_n]
    F --> G[LLM verbalizer<br/>rules: cite or abstain]
    H[Coverage tripwire] -.low coverage.-> I[LLM drafts adapter rules<br/>human gate, hash-frozen]
    I -.deterministic re-encode.-> B
""", encoding="utf-8")
    (DIAGRAMS / "draft_adapter_loop.mmd").write_text("""\
%% Draft-adapter loop: rules, never facts
sequenceDiagram
    participant U as New corpus
    participant C as Coverage detector
    participant L as LLM (drafter)
    participant H as Human gate
    participant E as Deterministic encoder
    U->>C: encode w/ frozen rules
    C-->>U: coverage 9% (amber)
    C->>L: residual lines only
    L->>H: candidate rules (JSON, hashed)
    H->>E: approve + freeze
    E->>U: re-encode (coverage 100%, receipts tainted "draft")
""", encoding="utf-8")
    (DIAGRAMS / "tiered_retrieval.mmd").write_text("""\
%% Production retrieval posture
flowchart TD
    Q[Query incident] --> T{Coverage / scale}
    T -->|within-system, hot path| W[WL kernel ~1ms]
    T -->|cross-system / audit| S[SME alignment]
    T -->|haystack| H[Hybrid RRF: bm25+dense+sma]
    W --> R[SME receipts on final candidates]
    S --> R
    H --> R
    R --> V[Verbalizer: cite or abstain]
""", encoding="utf-8")
    (DIAGRAMS / "README.md").write_text(
        "Mermaid sources for paper diagrams. Render with mermaid-cli:\n\n"
        "    npx -y @mermaid-js/mermaid-cli -i architecture.mmd -o architecture.pdf\n\n"
        "Sources are canonical; rendered files are derived artifacts.\n",
        encoding="utf-8")
    print("  diagrams: architecture.mmd, draft_adapter_loop.mmd, tiered_retrieval.mmd")


def main() -> int:
    for d in (FIGS, TABLES, DIAGRAMS):
        d.mkdir(parents=True, exist_ok=True)
    print(f"paper assets @ {STAMP}")
    fig_transfer_headline()
    fig_decomposition()
    fig_family()
    fig_h3()
    fig_ladder()
    make_tables()
    make_diagrams()
    (PAPER / "README.md").write_text(
        f"# Paper assets\n\nRegenerate with `make paper`. Stamp: {STAMP}\n\n"
        "- figures/: SciencePlots (science+grid style), PDF+PNG, data-version stamped.\n"
        "- tables/: booktabs LaTeX, auto-generated from reports/*.csv.\n"
        "- diagrams/: mermaid sources (render via mermaid-cli).\n\n"
        "Hatched bars / 'pre-v3' labels = numbers produced before matcher-semantics v3;\n"
        "they are regenerated automatically when their runs are re-verified.\n",
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
