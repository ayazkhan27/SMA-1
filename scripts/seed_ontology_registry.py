"""Seed a multi-domain OntologyRegistry + DomainRouter from configs/ontologies.json.

This is the "more than one ontology" entrypoint: it registers every curated
golden ontology that is present on disk, mounts each as an SMA is-a lattice +
higher-order relations, and demonstrates ACROSS-ecosystem routing (HP: -> hpo,
GO: -> go, T -> attack). It does NOT merge everything into one graph — that is
the deliberate non-goal from docs/PAPER_SPINE.md sec 7 (merge WITHIN an aligned
ecosystem, route ACROSS).

    python3 scripts/seed_ontology_registry.py            # report
    from scripts.seed_ontology_registry import build_registry
    reg, router = build_registry()                       # programmatic use
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from sma.ontology import DomainRouter, OntologyRegistry, load_attack_stix, mount

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/ontologies.json"


def _entries():
    spec = json.loads(MANIFEST.read_text())
    for eco in spec["ecosystems"].values():
        for o in eco["ontologies"]:
            yield o


def build_registry(require_present: bool = True):
    """Register + return (OntologyRegistry, DomainRouter) for on-disk ontologies."""
    reg = OntologyRegistry()
    router = DomainRouter(reg)
    for o in _entries():
        path = ROOT / o["path"]
        if require_present and not path.exists():
            continue
        # ATT&CK is STIX, not OBO/OWL: register a pre-mounted entry directly.
        if o["name"] == "attack":
            reg.register(o["name"], str(path), fmt="obo", version=o.get("version", ""))
        else:
            reg.register(o["name"], str(path), version=o.get("version", ""))
        router.register_prefix(o["prefix"], o["name"])
        router.register_domain(o["domain"], o["name"])
    return reg, router


def main():
    reg, router = build_registry()
    present = [e.name for e in reg.list()]
    print(f"=== ontology registry: {len(present)} on-disk ontologies ===\n")
    print(f"{'name':8}{'domain':10}{'terms':>8}{'is_a':>8}{'typed':>8}{'load(s)':>9}  version")
    for entry in reg.list():
        path = pathlib.Path(entry.path)
        t = time.perf_counter()
        if entry.name == "attack":
            g = load_attack_stix(str(path), name="attack"); mount(g)
        else:
            mo = reg.get(entry.name); g = mo.graph
        isa = sum(1 for _ in g.is_a_edges()); typed = sum(1 for _ in g.typed_relations())
        dom = next((o["domain"] for o in _entries() if o["name"] == entry.name), "")
        print(f"{entry.name:8}{dom:10}{len(g.active_terms()):8d}{isa:8d}{typed:8d}"
              f"{time.perf_counter()-t:9.2f}  {g.version}")

    print("\n=== ACROSS-ecosystem routing (no merge) ===")
    for term_ids, label in [
        (["HP:0001250", "HP:0001263"], "phenotypes"),
        (["GO:0008150", "GO:0003674"], "gene-function"),
        (["MONDO:0005148"], "disease"),
        (["T1059", "T1059.001"], "attack-techniques"),
        (["HP:0001250", "GO:0008150", "T1059"], "mixed (3 ecosystems)"),
    ]:
        print(f"  {label:24} {term_ids}  ->  {router.route(term_ids=term_ids) or '(no match)'}")

    print("\nRule: merge WITHIN an ecosystem (OBO via BFO/RO), route ACROSS. "
          "Each ontology is mounted separately; queries are routed, never unioned.")


if __name__ == "__main__":
    main()
