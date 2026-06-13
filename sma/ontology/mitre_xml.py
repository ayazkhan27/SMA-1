"""Loaders for MITRE CAPEC (attack patterns) and CWE (software weaknesses) XML.

Both ship as a flat list of entries with explicit ``ChildOf`` relations that form
a deep is-a hierarchy, plus other typed relations (CanPrecede/CanFollow/PeerOf,
and CAPEC->CWE ``exploits`` links). These enrich the cyber domain beyond ATT&CK's
shallow tactic/technique tree with real subsumption depth and cross-links.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

from .graph import OntologyGraph, Term

_KINDS = {
    # kind: (entry_tag, id_prefix, related_tag, id_attr)
    "capec": ("Attack_Pattern", "CAPEC", "Related_Attack_Pattern", "CAPEC_ID"),
    "cwe": ("Weakness", "CWE", "Related_Weakness", "CWE_ID"),
}


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def load_mitre_xml(path: str, kind: str, name: str = "") -> OntologyGraph:
    entry_tag, prefix, related_tag, id_attr = _KINDS[kind]
    name = name or kind
    tree = ET.parse(path)
    terms: dict[str, Term] = {}
    version = tree.getroot().get("Version", "")
    for elem in tree.getroot().iter():
        if _local(elem.tag) != entry_tag:
            continue
        eid = elem.get("ID")
        if not eid:
            continue
        tid = f"{prefix}-{eid}"
        obsolete = (elem.get("Status", "") or "").lower() in ("deprecated", "obsolete")
        parents: list[str] = []
        relations: list[tuple[str, str]] = []
        for rel in elem.iter():
            rlocal = _local(rel.tag)
            if rlocal == related_tag:
                nature = rel.get("Nature", "")
                target = rel.get(id_attr)
                if not target:
                    continue
                if nature == "ChildOf":
                    parents.append(f"{prefix}-{target}")
                elif nature:
                    relations.append((nature, f"{prefix}-{target}"))
            elif rlocal == "Related_Weakness" and kind == "capec":
                cwe = rel.get("CWE_ID")
                if cwe:
                    relations.append(("exploits", f"CWE-{cwe}"))
        terms[tid] = Term(id=tid, name=elem.get("Name", ""),
                          parents=tuple(dict.fromkeys(parents)),
                          relations=tuple(dict.fromkeys(relations)), obsolete=obsolete)
    return OntologyGraph(name=name, version=version, terms=terms)


def load_capec(path: str, name: str = "capec") -> OntologyGraph:
    return load_mitre_xml(path, "capec", name=name)


def load_cwe(path: str, name: str = "cwe") -> OntologyGraph:
    return load_mitre_xml(path, "cwe", name=name)
