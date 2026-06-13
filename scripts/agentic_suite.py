"""Driver for the one-shot agentic ontology suite (memory-swap benchmark).

Assembles all six memories (SMA + the enterprise-RAG/KG gauntlet) over an arm,
runs ``run_oneshot``, prints a per-memory table (tail top-1/5/10 on the all +
rare slices, risk-coverage AURC, novelty F1) plus the SMA-vs-best-RAG paired
bootstrap, and writes ``reports/confirmatory/agentic_<arm>.csv``.

  python3 scripts/agentic_suite.py --arm medicine
  python3 scripts/agentic_suite.py --arm medicine --fast   # skip slow reranker
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sma.eval.agentic import (
    BM25Memory,
    DenseMemory,
    HippoMemory,
    HybridRerankMemory,
    HybridRRFMemory,
    SmaMemory,
    run_oneshot,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/confirmatory"

ARMS = {"medicine": "sma.eval.agentic.arms.medicine"}


def build_memories(mounted, *, fast: bool) -> list:
    """Assemble the six (or five, under ``--fast``) memories for one arm."""
    bm25 = BM25Memory()
    dense = DenseMemory()
    hybrid = HybridRRFMemory(bm25, dense)
    memories = [SmaMemory(mounted), bm25, dense, hybrid, HippoMemory()]
    if not fast:
        memories.append(HybridRerankMemory(hybrid))
    return memories


def _fmt(slice_d: dict) -> str:
    return f"{slice_d['all']:.3f}/{slice_d['rare']:.3f}"


def print_table(result: dict) -> None:
    """Print the per-memory tail/AURC/novelty table and the primary verdict."""
    print(
        f"\n===== AGENTIC SUITE [{result['arm']}] "
        f"(n_all={result['n_all']} n_rare={result['n_rare']} n_novel={result['n_novel']}) ====="
    )
    print(
        f"{'memory':<16}{'t1 all/rare':<16}{'t5 all/rare':<16}"
        f"{'t10 all/rare':<16}{'AURC':<8}{'novF1':<8}"
    )
    for m in result["memories"]:
        pm = result["per_memory"][m]
        t = pm["tail"]
        print(
            f"{m:<16}{_fmt(t['top1']):<16}{_fmt(t['top5']):<16}"
            f"{_fmt(t['top10']):<16}{pm['aurc']:<8.3f}{pm['novelty_f1']:<8.3f}"
        )

    pr = result["primary"]
    if pr is None:
        print("\nprimary: n/a (need SMA + >=1 enterprise memory)")
        return
    verdict = "WIN" if (pr["p_value"] < 0.05 and pr["delta_top5"] > 0) else "parity/null"
    print(
        f"\nprimary: SMA vs best-RAG ({pr['best_enterprise']}) on tail top-5  "
        f"delta={pr['delta_top5']:+.4f} "
        f"CI=[{pr['ci_low']:.4f},{pr['ci_high']:.4f}] "
        f"p={pr['p_value']:.4f} cliffs={pr['cliffs']:.3f}  -> {verdict}"
    )


def write_csv(result: dict, arm: str) -> pathlib.Path:
    """Write a per-memory CSV row file and return its path."""
    OUT.mkdir(parents=True, exist_ok=True)
    pr = result["primary"] or {}
    rows = []
    for m in result["memories"]:
        pm = result["per_memory"][m]
        t = pm["tail"]
        rows.append(
            {
                "arm": arm,
                "memory": m,
                "n_all": result["n_all"],
                "n_rare": result["n_rare"],
                "n_novel": result["n_novel"],
                "t1_all": f"{t['top1']['all']:.4f}",
                "t1_rare": f"{t['top1']['rare']:.4f}",
                "t5_all": f"{t['top5']['all']:.4f}",
                "t5_rare": f"{t['top5']['rare']:.4f}",
                "t10_all": f"{t['top10']['all']:.4f}",
                "t10_rare": f"{t['top10']['rare']:.4f}",
                "aurc": f"{pm['aurc']:.4f}",
                "novelty_f1": f"{pm['novelty_f1']:.4f}",
                "best_enterprise": pr.get("best_enterprise", ""),
                "primary_delta_t5": f"{pr['delta_top5']:.4f}" if pr else "",
                "primary_ci_low": f"{pr['ci_low']:.4f}" if pr else "",
                "primary_ci_high": f"{pr['ci_high']:.4f}" if pr else "",
                "primary_p": f"{pr['p_value']:.4f}" if pr else "",
                "primary_cliffs": f"{pr['cliffs']:.4f}" if pr else "",
            }
        )
    path = OUT / f"agentic_{arm}.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="medicine", choices=sorted(ARMS))
    ap.add_argument("--n-index", type=int, default=2000)
    ap.add_argument("--n-query", type=int, default=120)
    ap.add_argument("--fast", action="store_true", help="skip the slow reranker memory")
    args = ap.parse_args()

    import importlib

    arm_mod = importlib.import_module(ARMS[args.arm])
    print(f"########## agentic arm: {args.arm} ##########", flush=True)
    mounted, records = arm_mod.load()
    print(f"  {len(records)} entities loaded; assembling memories...", flush=True)

    memories = build_memories(mounted, fast=args.fast)
    result = run_oneshot(
        args.arm,
        mounted,
        records,
        memories,
        n_index=args.n_index,
        n_query=args.n_query,
    )
    print_table(result)
    path = write_csv(result, args.arm)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
