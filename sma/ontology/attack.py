"""Load MITRE ATT&CK (STIX 2.1 JSON) into the normalized :class:`OntologyGraph`.

ATT&CK ships as a STIX bundle (``mitre/cti`` ``enterprise-attack.json``), not
OBO/OWL, so it needs a dedicated parser. It maps cleanly onto the same shape the
rest of the ontology package consumes:

* ``attack-pattern`` objects become technique terms, keyed by their ATT&CK
  ``external_id`` (e.g. ``"T1059"`` or sub-technique ``"T1059.001"``).
* ``x-mitre-tactic`` objects become tactic terms, keyed by their ``shortname``.
* A sub-technique ``T1059.001`` gets is_a parent ``T1059`` (split on ``"."``);
  this is corroborated by ``relationship`` objects of type ``subtechnique-of``.
* A technique's ``kill_chain_phases`` and STIX ``uses``/``mitigates``
  relationships become typed relations between the mapped external ids.

Revoked or ``x_mitre_deprecated`` objects are marked obsolete.
"""

from __future__ import annotations

import json
from pathlib import Path

from .graph import OntologyGraph, Term

#: ATT&CK download URL (kept here so the demo can surface it without fetching).
ATTACK_STIX_URL = (
    "https://raw.githubusercontent.com/mitre/cti/master/"
    "enterprise-attack/enterprise-attack.json"
)


def _external_id(obj: dict) -> str:
    """Return the ATT&CK external_id (e.g. ``T1059``) for a STIX object, or ``""``."""
    for ref in obj.get("external_references", ()):
        if ref.get("source_name") == "mitre-attack" and ref.get("external_id"):
            return ref["external_id"]
    return ""


def _is_obsolete(obj: dict) -> bool:
    """True if the STIX object is revoked or marked deprecated."""
    return bool(obj.get("revoked")) or bool(obj.get("x_mitre_deprecated"))


def load_attack_stix(path: str, name: str = "attack") -> OntologyGraph:
    """Parse an ATT&CK STIX 2.1 bundle into an :class:`OntologyGraph`.

    Techniques (``attack-pattern``) and tactics (``x-mitre-tactic``) become
    terms; sub-technique is_a edges, kill-chain ``accomplishes`` links, and
    ``uses``/``mitigates`` relationships become parents/typed relations between
    the resolved external ids.
    """
    with open(path, "r", encoding="utf-8") as handle:
        bundle = json.load(handle)

    version = str(bundle.get("spec_version", "") or "")
    objects = bundle.get("objects", [])

    terms: dict[str, Term] = {}
    # STIX object 'id' -> our term id (external_id / tactic shortname), so that
    # 'relationship' objects (which reference STIX ids) can resolve endpoints.
    stix_to_term: dict[str, str] = {}
    # Accumulate parents/relations per term id before constructing Term records.
    parents: dict[str, set[str]] = {}
    relations: dict[str, set[tuple[str, str]]] = {}
    obsolete: dict[str, bool] = {}
    names: dict[str, str] = {}

    # --- First pass: collect technique + tactic terms. --------------------- #
    for obj in objects:
        otype = obj.get("type")
        if otype == "attack-pattern":
            tid = _external_id(obj)
            if not tid:
                continue
            stix_to_term[obj.get("id", "")] = tid
            names[tid] = obj.get("name", "")
            obsolete[tid] = _is_obsolete(obj)
            parents.setdefault(tid, set())
            relations.setdefault(tid, set())
            # Sub-technique is_a parent derived by splitting the id on ".".
            if "." in tid:
                parents[tid].add(tid.split(".", 1)[0])
            # kill_chain_phases -> ("accomplishes", tactic_shortname)
            for phase in obj.get("kill_chain_phases", ()):
                if phase.get("kill_chain_name") == "mitre-attack":
                    pname = phase.get("phase_name")
                    if pname:
                        relations[tid].add(("accomplishes", pname))
        elif otype == "x-mitre-tactic":
            short = obj.get("x_mitre_shortname") or _external_id(obj)
            if not short:
                continue
            stix_to_term[obj.get("id", "")] = short
            names[short] = obj.get("name", "")
            obsolete[short] = _is_obsolete(obj)
            parents.setdefault(short, set())
            relations.setdefault(short, set())

    # --- Second pass: STIX relationship objects. --------------------------- #
    for obj in objects:
        if obj.get("type") != "relationship":
            continue
        if _is_obsolete(obj):
            continue
        rtype = obj.get("relationship_type")
        src = stix_to_term.get(obj.get("source_ref", ""))
        tgt = stix_to_term.get(obj.get("target_ref", ""))
        if not src or not tgt:
            continue
        if rtype == "subtechnique-of":
            # Corroborates (and is the source of truth for) the is_a edge.
            parents.setdefault(src, set()).add(tgt)
        elif rtype in ("uses", "mitigates"):
            relations.setdefault(src, set()).add((rtype, tgt))

    # --- Materialize Term records (parents/relations to resolvable ids). --- #
    for tid in names:
        ps = tuple(sorted(p for p in parents.get(tid, ()) if p in names))
        rs = tuple(sorted(
            (rel, obj_id) for rel, obj_id in relations.get(tid, ())
            if obj_id in names
        ))
        terms[tid] = Term(
            id=tid,
            name=names[tid],
            parents=ps,
            relations=rs,
            obsolete=obsolete.get(tid, False),
        )

    if not name:
        name = Path(path).stem
    return OntologyGraph(name=name, version=version, terms=terms)
