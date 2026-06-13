"""Demonstrate the SMA ontology mount path on MITRE ATT&CK (a second domain).

This proves the *same* ``load -> mount -> build_index -> retrieve`` pipeline that
serves the rare-disease (HPO) work also serves an unrelated golden ontology
(enterprise security techniques), supporting the "across fields" claim.

The script never touches the network. If the ATT&CK STIX bundle is absent it
prints the download URL and exits 0.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sma.ontology.attack import ATTACK_STIX_URL, load_attack_stix
from sma.ontology.mount import mount

DATA = pathlib.Path("data/raw/attack/enterprise-attack.json")


def main() -> int:
    if not DATA.exists():
        print(f"ATT&CK STIX bundle not found at: {DATA}")
        print("Download it (not fetched automatically) from:")
        print(f"  {ATTACK_STIX_URL}")
        print(f"  e.g.  mkdir -p {DATA.parent} && curl -L -o {DATA} \\")
        print(f"        {ATTACK_STIX_URL}")
        return 0

    graph = load_attack_stix(str(DATA), name="attack")
    mounted = mount(graph)

    n_terms = len(graph.terms)
    n_active = len(graph.active_terms())
    n_techniques = sum(1 for t in graph.active_terms().values() if t.id.startswith("T"))
    n_isa = sum(1 for _ in graph.is_a_edges())
    n_rels = sum(1 for _ in graph.typed_relations())

    print(f"ATT&CK loaded: version={graph.version!r}")
    print(f"  terms (total):        {n_terms}")
    print(f"  terms (active):       {n_active}")
    print(f"  techniques (active):  {n_techniques}")
    print(f"  is_a edges:           {n_isa}")
    print(f"  typed relations:      {n_rels}")

    # Build an index over every active technique as a singleton "signature"
    # record, then query with a small "incident" of co-occurring techniques.
    records = [
        (tid, [tid], {"name": term.name})
        for tid, term in graph.active_terms().items()
        if tid.startswith("T")
    ]
    if not records:
        print("No techniques available to index; skipping retrieval demo.")
        return 0

    index = mounted.build_index(records)

    # A handful of common adversary techniques as an incident query. Fall back
    # to whatever ids exist if these specific ones are absent.
    wanted = ["T1059", "T1059.001", "T1566", "T1486"]
    incident = [t for t in wanted if t in graph.active_terms()]
    if not incident:
        incident = [r[0] for r in records[:3]]

    print(f"\nQuery incident techniques: {incident}")
    query = mounted.build_case(incident)
    results = index.retrieve(query, k=10, shortlist=80, fac_budget=40)

    print("Nearest techniques:")
    for r in results[:10]:
        key = index.key_of.get(r.case_id, r.case_id)
        term = graph.terms.get(key)
        label = term.name if term else ""
        print(f"  {key:<14} {label}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
