"""Driver for the Phase 5 LLM-QA "trustworthy specialist" harness (prereg v2).

Holds the LLM and prompt FIXED and swaps only the retrieval memory
(``none`` / ``dense`` / ``sma``). For each condition it (1) calibrates the
cite-or-abstain threshold on a DISJOINT calibration split — retrieval-only, no
LLM spend — by maximising Youden's J on the raw grounding score, then (2) runs
the registered medicine question pools (answerable + held-out, the latter both
out-of-knowledge and novel) through :class:`QAAgent`, computes the trustworthy-QA
axes (accuracy, citation-faithfulness, abstention / risk-coverage, threshold-free
grounding-AUROC, and novelty recall / precision / F1), prints a table, and writes
``reports/confirmatory/qa_<memory>.csv`` (per-item rows) +
``qa_<memory>_summary.csv``.

  # tiny, NO DeepSeek spend (mock LLM, pilot pool sizes):
  python3 scripts/agentic_qa.py --memory sma --mock --pilot

  # full run with the real LLM (DeepSeek; costs money — only without --mock):
  python3 scripts/agentic_qa.py --memory sma --n-answerable 120 --n-held 120

The closed-book condition (``--memory none``) has no retrieval, so its
citation-faithfulness and grounding-AUROC cells are N/A (printed/written as
``NA``) and it has no calibrated threshold (LLM-only abstention), as registered.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sma.eval.agentic import DenseMemory, Query, SmaMemory
from sma.eval.agentic_qa import (
    QAAgent,
    abstention,
    accuracy,
    citation_faithfulness,
    grounding_auroc,
    novelty_f1,
    novelty_recall,
)
from sma.eval.agentic_qa.agent import MockLLM
from sma.eval.agentic_qa.metrics import auroc
from sma.eval.agentic_qa.pools import build_pools
from sma.ontology import load_obo, mount

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/confirmatory"
HP_OBO = ROOT / "data/raw/hpo/hp.obo"
HPOA = ROOT / "data/raw/hpo/phenotype.hpoa"

# Per-item result fields (order = the per-item CSV column order).
RESULT_FIELDS = (
    "gold_id",
    "gold_name",
    "answerable",
    "novel",
    "abstained",
    "pred_id",
    "answer",
    "novelty_flag",
    "confidence",
    "grounding_score",
)


def build_memory(name: str, mounted, index_items: list):
    """Assemble and index the chosen retrieval memory (or ``None`` closed-book)."""
    if name == "none":
        return None
    if name == "dense":
        memory = DenseMemory()
    elif name == "sma":
        memory = SmaMemory(mounted)
    else:  # pragma: no cover - argparse choices guard this
        raise ValueError(f"unknown memory {name!r}")
    memory.index(index_items)
    return memory


def make_llm(mock: bool):
    """The fixed LLM backend: a mock (no spend) or the real DeepSeek orchestrator."""
    if mock:
        return MockLLM()
    # Imported lazily so --mock runs never need the DeepSeek client / key.
    from sma.agent.llm import DeepSeekOrchestrator

    return DeepSeekOrchestrator()


def calibrate_threshold(memory, calib_items: list, k: int):
    """Fit the cite-or-abstain threshold on the DISJOINT calibration split.

    Retrieval-only — NO LLM spend. For each calibration case takes the top RAW
    grounding score, then picks the threshold maximising Youden's J (TPR − FPR)
    for the rule "answer iff score ≥ t": answerable (indexed-gold) cases are the
    positive/should-answer class, held-out cases the negative/should-abstain
    class. Candidate thresholds are midpoints between observed scores plus
    below-min / above-max sentinels, so the operating point generalises to the
    test split rather than memorising a calibration value. Returns
    ``(threshold, calib_auroc)``, or ``(None, None)`` when a class is empty (no
    calibration possible → the agent runs with LLM-only abstention).
    """
    pos: list[float] = []
    neg: list[float] = []
    for it in calib_items:
        r = memory.retrieve(Query(it.case_terms, it.case_text), k)
        score = r[0].score if r else 0.0
        (pos if it.answerable else neg).append(score)
    if not pos or not neg:
        return None, None

    observed = sorted(set(pos + neg))
    candidates = [observed[0] - 1.0]
    candidates += [(observed[i] + observed[i + 1]) / 2 for i in range(len(observed) - 1)]
    candidates += [observed[-1] + 1.0]

    best_t, best_j = candidates[0], -2.0
    for t in candidates:
        tpr = sum(1 for s in pos if s >= t) / len(pos)
        fpr = sum(1 for s in neg if s >= t) / len(neg)
        j = tpr - fpr
        if j > best_j:
            best_j, best_t = j, t
    return best_t, auroc(pos, neg)


def run(
    memory_name: str,
    *,
    mock: bool,
    n_answerable: int,
    n_held: int,
    k: int,
    n_index: int = 1500,
    n_calib: int = 60,
) -> dict:
    """Build pools + memory, calibrate the gate, run the agent, compute metrics."""
    print(f"########## LLM-QA: memory={memory_name} mock={mock} ##########", flush=True)
    mounted = mount(load_obo(str(HP_OBO), name="hpo"))
    pools = build_pools(
        mounted,
        str(HPOA),
        n_answerable=n_answerable,
        n_held=n_held,
        n_index=n_index,
        n_calib=n_calib,
    )
    index_items = pools["index_items"]
    calib_items = list(pools.get("calib_answerable", [])) + list(pools.get("calib_ook", []))
    print(
        f"  index={len(index_items)} answerable={len(pools['answerable'])} "
        f"ook/novel={len(pools['novel'])} calib={len(calib_items)}; assembling memory...",
        flush=True,
    )

    key_to_name = {it.key: it.meta.get("name", it.key) for it in index_items}
    key_to_terms = {it.key: it.term_ids for it in index_items}

    memory = build_memory(memory_name, mounted, index_items)

    # Calibrate the cite-or-abstain threshold on the DISJOINT calibration split
    # (retrieval-only, no LLM spend). Closed-book has no memory -> no gate; falls
    # back to LLM-only abstention.
    score_threshold, calib_auroc = (None, None)
    if memory is not None and calib_items:
        score_threshold, calib_auroc = calibrate_threshold(memory, calib_items, k)
        if score_threshold is not None:
            print(
                f"  calibrated score_threshold={score_threshold:.4f} "
                f"(calib AUROC {calib_auroc:.3f}, n={len(calib_items)})",
                flush=True,
            )
        else:
            print("  calibration skipped (a calibration class was empty)", flush=True)

    agent = QAAgent(
        make_llm(mock),
        memory,
        key_to_name=key_to_name,
        key_to_terms=key_to_terms,
        k=k,
        score_threshold=score_threshold,
    )

    # answerable + out-of-knowledge are disjoint pools; "ook" IS "novel" (held-out
    # cases are both unanswerable and novel), so it is run ONCE under its results.
    items = pools["answerable"] + pools["novel"]
    results = [agent.answer(it) for it in items]

    metrics = compute_metrics(results)
    return {
        "memory": memory_name,
        "results": results,
        "metrics": metrics,
        "score_threshold": score_threshold,
        "calib_auroc": calib_auroc,
    }


def compute_metrics(results: list[dict]) -> dict:
    """The pre-registered trustworthy-QA axes over the per-item results."""
    abst = abstention(results)
    nf1 = novelty_f1(results)
    return {
        "accuracy": accuracy(results),
        "citation_faithfulness": citation_faithfulness(results),  # None == N/A
        "abstain_recall": abst["abstain_recall"],
        "false_abstain": abst["false_abstain"],
        "selective_accuracy": abst["selective_accuracy"],
        "aurc": abst["aurc"],
        # Threshold-free known-vs-unknown discrimination of the raw grounding
        # score (None == N/A for closed-book, which has no retrieval).
        "grounding_auroc": grounding_auroc(results),
        "novelty_recall": novelty_recall(results),
        "novelty_precision": nf1["precision"],
        "novelty_f1": nf1["f1"],
        "novelty_fpr": nf1["fpr"],
    }


def _fmt_cf(value) -> str:
    """Citation-faithfulness is ``NA`` for closed-book (no retrieval), else 3dp."""
    return "NA" if value is None else f"{value:.3f}"


def print_table(run_result: dict) -> None:
    """Print the per-axis trustworthy-QA table for one memory condition."""
    m = run_result["metrics"]
    thr = run_result.get("score_threshold")
    print(f"\n===== TRUSTWORTHY-QA [{run_result['memory']}] =====")
    if thr is not None:
        print(f"  score_threshold      : {thr:.4f}  (calibrated, retrieval-only)")
    print(f"  accuracy             : {m['accuracy']:.3f}")
    print(f"  citation_faithful    : {_fmt_cf(m['citation_faithfulness'])}")
    print(f"  abstain_recall       : {m['abstain_recall']:.3f}")
    print(f"  false_abstain        : {m['false_abstain']:.3f}")
    print(f"  selective_accuracy   : {m['selective_accuracy']:.3f}")
    print(f"  abstention_aurc      : {m['aurc']:.3f}  (lower is better)")
    print(f"  grounding_auroc      : {_fmt_cf(m['grounding_auroc'])}  (known vs unknown)")
    print(f"  novelty_recall       : {m['novelty_recall']:.3f}")
    print(f"  novelty_precision    : {m['novelty_precision']:.3f}")
    print(f"  novelty_f1           : {m['novelty_f1']:.3f}")
    print(f"  novelty_fpr          : {m['novelty_fpr']:.3f}  (false flags on answerable)")


def write_csvs(run_result: dict) -> tuple[pathlib.Path, pathlib.Path]:
    """Write the per-item rows + the one-row summary; return both paths."""
    OUT.mkdir(parents=True, exist_ok=True)
    memory = run_result["memory"]

    per_item_path = OUT / f"qa_{memory}.csv"
    with per_item_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=("memory", *RESULT_FIELDS))
        w.writeheader()
        for r in run_result["results"]:
            w.writerow({"memory": memory, **{f: r[f] for f in RESULT_FIELDS}})

    m = run_result["metrics"]
    thr = run_result.get("score_threshold")
    summary_path = OUT / f"qa_{memory}_summary.csv"
    with summary_path.open("w", newline="") as fh:
        fields = [
            "memory",
            "score_threshold",
            "accuracy",
            "citation_faithfulness",
            "abstain_recall",
            "false_abstain",
            "selective_accuracy",
            "aurc",
            "grounding_auroc",
            "novelty_recall",
            "novelty_precision",
            "novelty_f1",
            "novelty_fpr",
        ]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerow(
            {
                "memory": memory,
                "score_threshold": "NA" if thr is None else f"{thr:.4f}",
                "accuracy": f"{m['accuracy']:.4f}",
                "citation_faithfulness": _fmt_cf(m["citation_faithfulness"]),
                "abstain_recall": f"{m['abstain_recall']:.4f}",
                "false_abstain": f"{m['false_abstain']:.4f}",
                "selective_accuracy": f"{m['selective_accuracy']:.4f}",
                "aurc": f"{m['aurc']:.4f}",
                "grounding_auroc": _fmt_cf(m["grounding_auroc"]),
                "novelty_recall": f"{m['novelty_recall']:.4f}",
                "novelty_precision": f"{m['novelty_precision']:.4f}",
                "novelty_f1": f"{m['novelty_f1']:.4f}",
                "novelty_fpr": f"{m['novelty_fpr']:.4f}",
            }
        )
    return per_item_path, summary_path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Phase 5 LLM-QA trustworthy-specialist harness.")
    ap.add_argument("--memory", choices=("none", "dense", "sma"), default="sma")
    ap.add_argument("--n-answerable", type=int, default=120)
    ap.add_argument("--n-held", type=int, default=120)
    ap.add_argument("--k", type=int, default=5, help="retrieval top-k")
    ap.add_argument(
        "--mock",
        action="store_true",
        help="use a deterministic MockLLM (NO DeepSeek spend)",
    )
    ap.add_argument(
        "--n-index", type=int, default=1500, help="diseases indexed (smaller = faster pilot)")
    ap.add_argument(
        "--n-calib",
        type=int,
        default=60,
        help="per-class calibration cases (retrieval-only, no LLM spend)",
    )
    ap.add_argument(
        "--pilot",
        action="store_true",
        help="small pool sizes for a quick smoke run (overrides -n flags to 8)",
    )
    args = ap.parse_args(argv)

    n_answerable, n_held = args.n_answerable, args.n_held
    n_index = args.n_index
    n_calib = args.n_calib
    if args.pilot:
        n_answerable = n_held = 8
        n_index = 200
        n_calib = 40

    run_result = run(
        args.memory,
        mock=args.mock,
        n_answerable=n_answerable,
        n_held=n_held,
        k=args.k,
        n_index=n_index,
        n_calib=n_calib,
    )
    print_table(run_result)
    per_item_path, summary_path = write_csvs(run_result)
    print(f"\nwrote {per_item_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
