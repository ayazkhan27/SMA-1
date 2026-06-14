#!/usr/bin/env python3
"""Computational cost/latency micro-benchmark: SMA vs BM25 vs dense (BGE).

One (domain, memory) per process so peak-RSS is isolated per arm. Measures index
build time, per-query p50/p95 latency, and peak resident memory. Appends a row to
reports/confirmatory/cost_latency.csv.

Usage: python scripts/cost_latency.py --arm medicine --memory sma
"""
from __future__ import annotations
import argparse, csv, importlib, pathlib, resource, statistics, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sma.eval.agentic import IndexItem, Query, SmaMemory, BM25Memory, DenseMemory

ARMS = {"medicine": "sma.eval.agentic.arms.medicine",
        "discovery": "sma.eval.agentic.arms.discovery",
        "finance": "sma.eval.agentic.arms.finance",
        "legal": "sma.eval.agentic.arms.legal",
        "cyber": "sma.eval.agentic.arms.cyber"}


def make_mem(kind, mounted):
    if kind == "sma":
        return SmaMemory(mounted)
    if kind == "bm25":
        return BM25Memory()
    if kind == "dense":
        return DenseMemory()
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="medicine", choices=list(ARMS))
    ap.add_argument("--memory", default="sma", choices=["sma", "bm25", "dense"])
    ap.add_argument("--n-index", type=int, default=2000)
    ap.add_argument("--n-query", type=int, default=200)
    args = ap.parse_args()

    mod = importlib.import_module(ARMS[args.arm])
    mounted, records = mod.load()
    graph = mounted.graph
    items = []
    for eid, terms in records.items():
        known = sorted(t for t in terms if t in graph.terms)
        if known:
            items.append(IndexItem(key=eid, term_ids=frozenset(known),
                                   text=" ".join(graph.terms[t].name or t for t in known),
                                   meta={"id": eid}))
    idx_items = items[:args.n_index]
    qry_items = items[args.n_index:args.n_index + args.n_query] or items[:args.n_query]

    mem = make_mem(args.memory, mounted)
    t0 = time.perf_counter()
    mem.index(idx_items)
    t_index = time.perf_counter() - t0

    lat = []
    for it in qry_items:
        q = Query(term_ids=it.term_ids, text=it.text)
        t0 = time.perf_counter()
        mem.retrieve(q, k=10)
        lat.append((time.perf_counter() - t0) * 1000.0)
    lat.sort()
    p50 = statistics.median(lat)
    p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))]
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    out = ROOT / "reports/confirmatory/cost_latency.csv"
    fields = ["arm", "memory", "n_index", "n_query", "index_build_s",
              "p50_ms", "p95_ms", "mean_ms", "peak_rss_mb"]
    row = {"arm": args.arm, "memory": args.memory, "n_index": len(idx_items),
           "n_query": len(qry_items), "index_build_s": round(t_index, 3),
           "p50_ms": round(p50, 3), "p95_ms": round(p95, 3),
           "mean_ms": round(statistics.mean(lat), 3), "peak_rss_mb": round(peak_mb, 1)}
    write_header = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(row, flush=True)


if __name__ == "__main__":
    main()
