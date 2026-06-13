"""Seed the 5-domain ontology registry + DomainRouter from configs/ontologies.json.

Loads + mounts every golden ontology present on disk (handling obo / owl /
owl_dir / stix formats), wires prefix- and domain-based routing, and demonstrates
ACROSS-ecosystem routing without merging. It does NOT build one omni-graph -- the
deliberate non-goal from docs/PAPER_SPINE.md sec 7 (merge WITHIN an aligned
ecosystem, route ACROSS).

    python3 scripts/seed_ontology_registry.py            # report
    from scripts.seed_ontology_registry import build_mounts
    mounts, router = build_mounts()                      # programmatic use
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from sma.ontology import (DomainRouter, OntologyRegistry, load_attack_stix,
                          load_cpc, load_obo, load_owl_dir, mount)

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/ontologies.json"


def _entries():
    return json.loads(MANIFEST.read_text())["ontologies"]


def _load_graph(o):
    path = ROOT / o["path"]
    fmt = o["fmt"]
    if fmt == "obo":
        return load_obo(str(path), name=o["name"])
    if fmt == "stix":
        return load_attack_stix(str(path), name=o["name"])
    if fmt == "owl_dir":
        return load_owl_dir(str(path), name=o["name"], pattern=o.get("pattern", "*.rdf"))
    if fmt == "cpc":
        return load_cpc(str(path), name=o["name"])
    raise ValueError(f"unknown fmt {fmt}")


def build_mounts(require_present: bool = True):
    """Return (mounts: dict[name->MountedOntology], router) for on-disk ontologies."""
    mounts = {}
    reg = OntologyRegistry()           # kept for single-file API parity / future use
    router = DomainRouter(reg)
    for o in _entries():
        path = ROOT / o["path"]
        if require_present and not path.exists():
            continue
        mounts[o["name"]] = mount(_load_graph(o))
        if o.get("prefix"):
            router.register_prefix(o["prefix"], o["name"])
        router.register_domain(o["domain"], o["name"])
    return mounts, router


def main():
    mounts, router = build_mounts()
    print(f"=== 5-domain ontology registry: {len(mounts)} ontologies on disk ===\n")
    print(f"{'name':8}{'domain':10}{'terms':>8}{'is_a':>8}{'typed':>8}{'load(s)':>9}  version")
    for o in _entries():
        if o["name"] not in mounts:
            print(f"{o['name']:8}{o['domain']:10}{'--- not on disk ---':>33}")
            continue
        t = time.perf_counter(); g = _load_graph(o)
        isa = sum(1 for _ in g.is_a_edges()); typed = sum(1 for _ in g.typed_relations())
        print(f"{o['name']:8}{o['domain']:10}{len(g.active_terms()):8d}{isa:8d}{typed:8d}"
              f"{time.perf_counter()-t:9.2f}  {g.version[:30]}")

    print("\n=== ACROSS-ecosystem routing (no merge) ===")
    for term_ids, domain, label in [
        (["HP:0001250", "HP:0001263"], None, "phenotypes"),
        (["GO:0008150"], None, "gene-function"),
        (["CHEBI:15377"], None, "chemical"),
        (["T1059", "T1059.001"], None, "attack-techniques"),
        (None, "legal", "legal (by domain)"),
        (None, "finance", "finance (by domain)"),
        (["HP:0001250", "GO:0008150", "T1059"], None, "mixed (3 ecosystems)"),
    ]:
        print(f"  {label:24} {term_ids or domain}  ->  {router.route(term_ids=term_ids, domain=domain) or '(no match)'}")

    print("\nRule: merge WITHIN an ecosystem (OBO via BFO/RO), route ACROSS. "
          "Each ontology mounted separately; queries routed, never unioned.")


if __name__ == "__main__":
    main()
