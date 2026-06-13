"""Driver for the Phase 5 LLM-QA "trustworthy specialist" harness (prereg v2).

Holds the LLM and prompt FIXED and swaps only the retrieval memory
(``none`` / ``dense`` / ``sma``), runs the three registered medicine question
pools (answerable / out-of-knowledge / novel) through :class:`QAAgent`, computes
the four trustworthy-QA axes (accuracy, citation-faithfulness, abstention /
risk-coverage, novelty-recall), prints a table, and writes
``reports/confirmatory/qa_<memory>.csv`` (per-item rows) +
``qa_<memory>_summary.csv``.

  # tiny, NO DeepSeek spend (mock LLM, pilot pool sizes):
  python3 scripts/agentic_qa.py --memory sma --mock --pilot

  # full run with the real LLM (DeepSeek; costs money — only without --mock):
  python3 scripts/agentic_qa.py --memory sma --n-answerable 120 --n-held 120

The closed-book condition (``--memory none``) has no retrieval, so its
citation-faithfulness cell is N/A (printed/written as ``NA``), as registered.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sma.eval.agentic import DenseMemory, SmaMemory
from sma.eval.agentic_qa import (
    QAAgent,
    abstention,
    accuracy,
    citation_faithfulness,
    novelty_recall,
)
from sma.eval.agentic_qa.agent import MockLLM
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


def run(memory_name: str, *, mock: bool, n_answerable: int, n_held: int, k: int, n_index: int = 1500) -> dict:
    """Build pools + memory, run every pool through the agent, compute metrics."""
    print(f"########## LLM-QA: memory={memory_name} mock={mock} ##########", flush=True)
    mounted = mount(load_obo(str(HP_OBO), name="hpo"))
    pools = build_pools(
        mounted,
        str(HPOA),
        n_answerable=n_answerable,
        n_held=n_held,
        n_index=n_index,
    )
    index_items = pools["index_items"]
    print(
        f"  index={len(index_items)} answerable={len(pools['answerable'])} "
        f"ook/novel={len(pools['novel'])}; assembling memory...",
        flush=True,
    )

    key_to_name = {it.key: it.meta.get("name", it.key) for it in index_items}
    key_to_terms = {it.key: it.term_ids for it in index_items}

    memory = build_memory(memory_name, mounted, index_items)
    agent = QAAgent(
        make_llm(mock),
        memory,
        key_to_name=key_to_name,
        key_to_terms=key_to_terms,
        k=k,
    )

    # answerable + out-of-knowledge are disjoint pools; "ook" IS "novel" (held-out
    # cases are both unanswerable and novel), so it is run ONCE under its results.
    items = pools["answerable"] + pools["novel"]
    results = [agent.answer(it) for it in items]

    metrics = compute_metrics(results)
    return {"memory": memory_name, "results": results, "metrics": metrics}


def compute_metrics(results: list[dict]) -> dict:
    """The four pre-registered trustworthy-QA axes over the per-item results."""
    abst = abstention(results)
    return {
        "accuracy": accuracy(results),
        "citation_faithfulness": citation_faithfulness(results),  # None == N/A
        "abstain_recall": abst["abstain_recall"],
        "false_abstain": abst["false_abstain"],
        "selective_accuracy": abst["selective_accuracy"],
        "aurc": abst["aurc"],
        "novelty_recall": novelty_recall(results),
    }


def _fmt_cf(value) -> str:
    """Citation-faithfulness is ``NA`` for closed-book (no retrieval), else 3dp."""
    return "NA" if value is None else f"{value:.3f}"


def print_table(run_result: dict) -> None:
    """Print the per-axis trustworthy-QA table for one memory condition."""
    m = run_result["metrics"]
    print(f"\n===== TRUSTWORTHY-QA [{run_result['memory']}] =====")
    print(f"  accuracy             : {m['accuracy']:.3f}")
    print(f"  citation_faithful    : {_fmt_cf(m['citation_faithfulness'])}")
    print(f"  abstain_recall       : {m['abstain_recall']:.3f}")
    print(f"  false_abstain        : {m['false_abstain']:.3f}")
    print(f"  selective_accuracy   : {m['selective_accuracy']:.3f}")
    print(f"  abstention_aurc      : {m['aurc']:.3f}  (lower is better)")
    print(f"  novelty_recall       : {m['novelty_recall']:.3f}")


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
    summary_path = OUT / f"qa_{memory}_summary.csv"
    with summary_path.open("w", newline="") as fh:
        fields = [
            "memory",
            "accuracy",
            "citation_faithfulness",
            "abstain_recall",
            "false_abstain",
            "selective_accuracy",
            "aurc",
            "novelty_recall",
        ]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerow(
            {
                "memory": memory,
                "accuracy": f"{m['accuracy']:.4f}",
                "citation_faithfulness": _fmt_cf(m["citation_faithfulness"]),
                "abstain_recall": f"{m['abstain_recall']:.4f}",
                "false_abstain": f"{m['false_abstain']:.4f}",
                "selective_accuracy": f"{m['selective_accuracy']:.4f}",
                "aurc": f"{m['aurc']:.4f}",
                "novelty_recall": f"{m['novelty_recall']:.4f}",
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
        "--pilot",
        action="store_true",
        help="small pool sizes for a quick smoke run (overrides -n flags to 8)",
    )
    args = ap.parse_args(argv)

    n_answerable, n_held = args.n_answerable, args.n_held
    n_index = args.n_index
    if args.pilot:
        n_answerable = n_held = 8
        n_index = 200

    run_result = run(
        args.memory,
        mock=args.mock,
        n_answerable=n_answerable,
        n_held=n_held,
        k=args.k,
        n_index=n_index,
    )
    print_table(run_result)
    per_item_path, summary_path = write_csvs(run_result)
    print(f"\nwrote {per_item_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
