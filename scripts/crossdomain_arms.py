"""Run all three 4b arms on ONE fixed split for a valid comparison:
generic structured  vs  LLM-drafted (saved rules)  vs  expert hand-designed.

  python3 scripts/crossdomain_arms.py --domain diabetes
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from sma.eval.crossdomain import evaluate_arm
from sma.encoders import get_encoder
from sma.ir.schema import make_case
from scripts.crossdomain_after import apply_rules

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_domain(domain):
    if domain == "diabetes":
        from sma.eval.diabetes import load_encounters, row_csv, row_text
        items = load_encounters(sample=1000, seed=7, balanced=True)
        return items, row_csv, row_text, ("early", "not")
    from sma.eval.ieee_cis import load_transactions, row_csv, row_text
    items = load_transactions(sample=1000, seed=7, balanced=True)
    return items, row_csv, row_text, ("fraud", "legit")


def main(domain):
    items, row_csv, row_text, labels = load_domain(domain)
    structured = get_encoder("structured")
    rules = json.loads((ROOT / f"reports/confirmatory/cd_{domain}_rules.json").read_text())["rules"]
    healthcare = get_encoder("healthcare") if domain == "diabetes" else None

    arms = {
        "before": lambda it: structured.encode(row_csv(it), format="csv").case,
        "after": lambda it: make_case(apply_rules(it.fields, rules), {"adapter": "draft"}),
    }
    if healthcare is not None:
        arms["expert"] = lambda it: healthcare.encode_record(it.fields)

    print(f"=== {domain}: three arms, identical split ===", flush=True)
    out = {}
    for phase, fn in arms.items():
        out[phase] = evaluate_arm(items, fn, row_text, labels, phase, domain)
    print("\n=== comparison (same split, baselines now identical across arms) ===")
    print(f"{'arm':<10}{'HO-dens':<10}{'SMA':<8}{'BM25':<8}{'Dense':<8}")
    for phase, r in out.items():
        print(f"{phase:<10}{r['ho']:<10.3f}{r['f1']['SMA']:<8.3f}{r['f1']['BM25']:<8.3f}{r['f1']['Dense']:<8.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="diabetes", choices=["diabetes", "ieee"])
    main(ap.parse_args().domain)
