"""Run the pre-registered multi-domain ontology benchmark suite (the gigatest).

  python3 scripts/bench_ontology_suite.py            # all runnable arms (A1,A2)
  python3 scripts/bench_ontology_suite.py --arm go   # one arm

Each arm goes through the IDENTICAL universal pipeline (sma.eval.ontology_bench).
Holm-Bonferroni is applied across the arm family. Writes a summary CSV. See
configs/preregistration_ontology.md.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from sma.eval.ontology_bench import run_arm
from sma.eval.stats import holm_bonferroni
from sma.ontology import load_obo, mount

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/confirmatory"


def load_hpo_records():
    rec: dict[str, set] = {}
    for line in (ROOT / "data/raw/hpo/phenotype.hpoa").open():
        if line.startswith(("#", "database_id")):
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) < 11 or p[10] != "P":
            continue
        rec.setdefault(p[0], set()).add(p[3])
    return rec


def load_go_records(aspect="P"):
    rec: dict[str, set] = {}
    for line in (ROOT / "data/raw/go/goa_human.gaf").open():
        if line.startswith("!"):
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) < 9 or p[8] != aspect:
            continue
        rec.setdefault(p[1], set()).add(p[4])   # protein -> GO term
    return rec


ARMS = {
    "hpo": ("A1-hpo-rare-disease", "data/raw/hpo/hp.obo", "hpo", load_hpo_records),
    "go": ("A2-go-gene-function", "data/raw/obo/go-basic.obo", "go", load_go_records),
}


def main(which):
    keys = list(ARMS) if which == "all" else [which]
    results = []
    for k in keys:
        label, obo, name, loader = ARMS[k]
        print(f"\n########## {label} ##########", flush=True)
        mo = mount(load_obo(str(ROOT / obo), name=name))
        records = loader()
        print(f"  {len(records)} entities loaded; mounting + running...", flush=True)
        results.append(run_arm(label, mo, records))

    pvals = {r["arm"]: r["p_value"] for r in results}
    holm = holm_bonferroni(pvals)
    print("\n===== SUITE SUMMARY (Holm across arms) =====")
    print(f"{'arm':<24}{'SMA t5':<9}{'best base t5':<14}{'delta':<10}{'p_holm':<9}{'verdict'}")
    rows = []
    for r in results:
        ph = holm[r["arm"]]
        base_t5 = r[r["best_baseline"]]["t5"]
        win = "WIN" if (ph < 0.05 and r["primary_delta_t5"] > 0) else "parity/null"
        print(f"{r['arm']:<24}{r['sma']['t5']:<9.3f}{base_t5:<14.3f}"
              f"{r['primary_delta_t5']:<+10.4f}{ph:<9.4f}{win}")
        rows.append({
            "arm": r["arm"], "n": r["n_queries"], "best_baseline": r["best_baseline"],
            "sma_t1": f"{r['sma']['t1']:.4f}", "sma_t5": f"{r['sma']['t5']:.4f}",
            "sma_t10": f"{r['sma']['t10']:.4f}", "sma_mrr": f"{r['sma_mrr']:.4f}",
            "base_t5": f"{base_t5:.4f}", "delta_t5": f"{r['primary_delta_t5']:.4f}",
            "ci_low": f"{r['ci_low']:.4f}", "ci_high": f"{r['ci_high']:.4f}",
            "p_value": f"{r['p_value']:.4f}", "p_holm": f"{ph:.4f}",
            "cliffs": f"{r['cliffs']:.4f}", "verdict": win,
        })
    confirmed = sum(1 for x in rows if x["verdict"] == "WIN")
    print(f"\nC3 (across fields): {confirmed} arm(s) confirm C1. "
          f"{'CONFIRMED' if confirmed >= 2 else 'NOT YET (need >=2)'}")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "ontology_suite.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {OUT/'ontology_suite.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="all", choices=["all", "hpo", "go"])
    main(ap.parse_args().arm)
