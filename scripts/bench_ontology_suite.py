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


def main(which, n_index=2000, n_query=120):
    keys = list(ARMS) if which == "all" else [which]
    results = []
    for k in keys:
        label, obo, name, loader = ARMS[k]
        print(f"\n########## {label} ##########", flush=True)
        mo = mount(load_obo(str(ROOT / obo), name=name))
        records = loader()
        print(f"  {len(records)} entities loaded; mounting + running...", flush=True)
        results.append(run_arm(label, mo, records, n_index=n_index, n_query=n_query))

    # Holm across arms, on each slice's primary p-value separately
    rows = []
    for slice_name in ("all", "rare"):
        pvals = {r["arm"]: r["slices"][slice_name]["p_value"] for r in results}
        holm = holm_bonferroni(pvals)
        print(f"\n===== SUITE SUMMARY [{slice_name} slice] (Holm across arms) =====")
        print(f"{'arm':<24}{'SMA t5':<9}{'best(base)':<18}{'delta':<10}{'p_holm':<9}{'verdict'}")
        for r in results:
            s = r["slices"][slice_name]; ph = holm[r["arm"]]
            base_t5 = s["metrics"][s["best_baseline"]]["t5"]
            blabel = s["best_baseline"]
            win = "WIN" if (ph < 0.05 and s["delta_t5"] > 0) else "parity/null"
            print(f"{r['arm']:<24}{s['metrics']['sma']['t5']:<9.3f}"
                  f"{f'{base_t5:.3f}({blabel})':<18}{s['delta_t5']:<+10.4f}{ph:<9.4f}{win}")
            rows.append({
                "arm": r["arm"], "slice": slice_name, "n": s["n"], "best_baseline": blabel,
                "sma_t1": f"{s['metrics']['sma']['t1']:.4f}", "sma_t5": f"{s['metrics']['sma']['t5']:.4f}",
                "sma_t10": f"{s['metrics']['sma']['t10']:.4f}",
                "dense_t5": f"{s['metrics']['dense']['t5']:.4f}", "hippo_t5": f"{s['metrics']['hippo']['t5']:.4f}",
                "phen_t5": f"{s['metrics']['phen']['t5']:.4f}", "jac_t5": f"{s['metrics']['jac']['t5']:.4f}",
                "base_t5": f"{base_t5:.4f}", "delta_t5": f"{s['delta_t5']:.4f}",
                "ci_low": f"{s['ci_low']:.4f}", "ci_high": f"{s['ci_high']:.4f}",
                "p_value": f"{s['p_value']:.4f}", "p_holm": f"{ph:.4f}",
                "cliffs": f"{s['cliffs']:.4f}", "verdict": win,
            })
        confirmed = sum(1 for r in results if holm[r["arm"]] < 0.05 and r["slices"][slice_name]["delta_t5"] > 0)
        print(f"  C3 [{slice_name}]: {confirmed} arm(s) confirm C1 vs REAL baselines. "
              f"{'CONFIRMED' if confirmed >= 2 else 'NOT YET (need >=2)'}")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "ontology_suite.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {OUT/'ontology_suite.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="all", choices=["all", "hpo", "go"])
    main(ap.parse_args().arm)
