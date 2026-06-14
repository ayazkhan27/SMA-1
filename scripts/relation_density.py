#!/usr/bin/env python3
"""Why don't higher-order typed relations help on the real benchmarks?

A case gains a higher-order statement only when BOTH endpoints of a typed relation
are co-present on the same entity (sma/ontology/mount.py:build_case). This probe
measures, per relation-bearing domain, how often that actually happens in the real
records: the mean number of higher-order statements per case and the fraction of
cases that carry any. If real cases rarely contain co-present related term-pairs,
the higher-order machinery has nothing to align — explaining the ablation null.

Output: reports/confirmatory/relation_density.csv
"""
from __future__ import annotations
import csv, importlib, pathlib, statistics, sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Relation-bearing arms only (HPO/CPC carry zero typed relations -> trivially 0).
ARMS = {"genomics": "sma.eval.agentic.arms.discovery",
        "cyber": "sma.eval.agentic.arms.cyber",
        "finance": "sma.eval.agentic.arms.finance"}


def main():
    rows = []
    for arm, path in ARMS.items():
        mounted, records = importlib.import_module(path).load()
        graph = mounted.graph
        rel_by_src: dict[str, set[str]] = defaultdict(set)
        n_typed = 0
        for s, _r, o in graph.typed_relations():
            rel_by_src[s].add(o); n_typed += 1
        ho_per_case, set_sizes = [], []
        for _e, terms in records.items():
            T = {t for t in terms if t in graph.terms}
            if not T:
                continue
            set_sizes.append(len(T))
            ho = sum(1 for s in T for o in rel_by_src.get(s, ()) if o in T)
            ho_per_case.append(ho)
        n = len(ho_per_case)
        frac_any = sum(1 for c in ho_per_case if c >= 1) / n if n else 0.0
        row = {"domain": arm, "n_cases": n,
               "typed_relations_in_ontology": n_typed,
               "mean_terms_per_case": round(statistics.mean(set_sizes), 2) if set_sizes else 0,
               "mean_higher_order_per_case": round(statistics.mean(ho_per_case), 4) if ho_per_case else 0,
               "frac_cases_with_any_higher_order": round(frac_any, 4)}
        rows.append(row)
        print(row, flush=True)
    out = ROOT / "reports/confirmatory/relation_density.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
