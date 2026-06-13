"""Discovery arm: gene-function retrieval over the Gene Ontology (GO).

Entities are human proteins; each is annotated with its set of GO biological-process
terms (gold, from the GOA GAF). The hard query is a partial/imprecise functional
profile; the task is to retrieve the right protein by its function signature. GO's
is-a tree is the ascension lattice and its part_of/regulates relations become
higher-order statements.
"""
from __future__ import annotations

import pathlib

from sma.ontology import load_obo, mount

ROOT = pathlib.Path(__file__).resolve().parents[4]
GO_OBO = ROOT / "data/raw/obo/go-basic.obo"
GAF = ROOT / "data/raw/go/goa_human.gaf"
ASPECT = "P"            # biological process
MIN_TERMS, MAX_TERMS = 7, 30


def load():
    mounted = mount(load_obo(str(GO_OBO), name="go"))
    known = mounted.graph.terms
    recs: dict[str, set[str]] = {}
    for line in GAF.open():
        if line.startswith("!"):
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) < 9 or p[8] != ASPECT:
            continue
        if p[4] in known:
            recs.setdefault(p[1], set()).add(p[4])      # protein -> GO term
    records = {k: v for k, v in recs.items() if MIN_TERMS <= len(v) <= MAX_TERMS}
    return mounted, records
