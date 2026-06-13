"""Finance arm: SEC filing classification over the US-GAAP reporting taxonomy.

Entities are SEC filings (accession numbers); each is annotated with the set of
US-GAAP concepts it reports (gold, from the SEC Financial Statement Data Sets'
num.txt). The hard query is a partial/imprecise set of reported concepts climbed
up the statement hierarchy plus noise; the task is to retrieve the filing by its
financial-reporting signature. The US-GAAP presentation hierarchy (statement
headers subsume line items) is the is-a lattice (FIBO, a schema, has no instance
corpus, so US-GAAP provides the financial entity->concept gold).
"""
from __future__ import annotations

import pathlib

from sma.ontology import mount
from sma.ontology.usgaap import load_usgaap

ROOT = pathlib.Path(__file__).resolve().parents[4]
USGAAP_DIR = ROOT / "data/raw/finance/usgaap"
NUM = ROOT / "data/raw/finance/sec_2024q1/num.txt"
MIN_TERMS, MAX_TERMS = 10, 40


def load():
    mounted = mount(load_usgaap(str(USGAAP_DIR), name="usgaap"))
    known = mounted.graph.terms
    recs: dict[str, set[str]] = {}
    with NUM.open(encoding="utf-8", errors="ignore") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        ia, it, iv = header.index("adsh"), header.index("tag"), header.index("version")
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) <= iv or not p[iv].startswith("us-gaap"):
                continue
            if p[it] in known:
                recs.setdefault(p[ia], set()).add(p[it])
    records = {k: v for k, v in recs.items() if MIN_TERMS <= len(v) <= MAX_TERMS}
    return mounted, records
