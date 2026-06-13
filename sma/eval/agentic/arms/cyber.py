"""Cyber arm: threat-group attribution over MITRE ATT&CK.

Entities are intrusion-sets (threat groups); each is annotated with the set of
ATT&CK techniques it is documented to USE (gold, from the STIX ``uses``
relationships). The hard query simulates a partial/imprecise incident (a few
observed TTPs, climbed up the sub-technique lattice, plus noise); the task is to
attribute it to the right group. ATT&CK's sub-technique tree is the is-a lattice.
"""
from __future__ import annotations

import json
import pathlib

from sma.ontology import load_attack_stix, mount

ROOT = pathlib.Path(__file__).resolve().parents[4]
STIX = ROOT / "data/raw/attack/enterprise-attack.json"
# Band matches the medicine arm: >=7 techniques to be non-trivial, <=30 to keep
# structure-mapping cases tractable (prolific groups with 100+ techniques blow up
# SME kernel enumeration). Excludes the most-documented groups; n reported.
MIN_TERMS, MAX_TERMS = 7, 30


def load():
    graph = load_attack_stix(str(STIX), name="attack")
    mounted = mount(graph)
    bundle = json.loads(STIX.read_text())
    objs = bundle["objects"]

    # stix-id -> ATT&CK external_id (e.g. "T1059.001"), for attack-patterns.
    ext: dict[str, str] = {}
    for o in objs:
        if o.get("type") == "attack-pattern":
            for ref in o.get("external_references", []):
                if ref.get("source_name") == "mitre-attack" and ref.get("external_id"):
                    ext[o["id"]] = ref["external_id"]
    groups = {o["id"]: o.get("name", o["id"])
              for o in objs if o.get("type") == "intrusion-set" and not o.get("revoked")}

    recs: dict[str, set[str]] = {}
    for o in objs:
        if o.get("type") == "relationship" and o.get("relationship_type") == "uses":
            s, t = o.get("source_ref"), o.get("target_ref")
            if s in groups and t in ext and ext[t] in graph.terms:
                recs.setdefault(groups[s], set()).add(ext[t])

    records = {g: ts for g, ts in recs.items() if MIN_TERMS <= len(ts) <= MAX_TERMS}
    return mounted, records
