"""Tests for sma.ontology.attack: STIX 2.1 -> OntologyGraph (no real data)."""

from __future__ import annotations

import json

import pytest

from sma.ontology.attack import load_attack_stix
from sma.ontology.loader import fid
from sma.ontology.mount import mount


@pytest.fixture
def stix_bundle() -> dict:
    """A tiny synthetic ATT&CK STIX bundle.

    Contents: 1 tactic, 1 technique (T1059), 1 sub-technique (T1059.001), a
    'uses' relationship, and one revoked technique (T9999).
    """
    return {
        "type": "bundle",
        "spec_version": "2.1",
        "objects": [
            {
                "type": "x-mitre-tactic",
                "id": "x-mitre-tactic--exec",
                "name": "Execution",
                "x_mitre_shortname": "execution",
            },
            {
                "type": "attack-pattern",
                "id": "attack-pattern--t1059",
                "name": "Command and Scripting Interpreter",
                "external_references": [
                    {"source_name": "mitre-attack", "external_id": "T1059"}
                ],
                "kill_chain_phases": [
                    {"kill_chain_name": "mitre-attack", "phase_name": "execution"}
                ],
            },
            {
                "type": "attack-pattern",
                "id": "attack-pattern--t1059-001",
                "name": "PowerShell",
                "external_references": [
                    {"source_name": "mitre-attack", "external_id": "T1059.001"}
                ],
                "kill_chain_phases": [
                    {"kill_chain_name": "mitre-attack", "phase_name": "execution"}
                ],
            },
            {
                "type": "attack-pattern",
                "id": "attack-pattern--t9999",
                "name": "Revoked Technique",
                "revoked": True,
                "external_references": [
                    {"source_name": "mitre-attack", "external_id": "T9999"}
                ],
            },
            {
                "type": "relationship",
                "id": "relationship--subof",
                "relationship_type": "subtechnique-of",
                "source_ref": "attack-pattern--t1059-001",
                "target_ref": "attack-pattern--t1059",
            },
            {
                "type": "relationship",
                "id": "relationship--uses",
                "relationship_type": "uses",
                "source_ref": "attack-pattern--t1059",
                "target_ref": "attack-pattern--t1059-001",
            },
        ],
    }


@pytest.fixture
def graph(tmp_path, stix_bundle):
    path = tmp_path / "mini-attack.json"
    path.write_text(json.dumps(stix_bundle), encoding="utf-8")
    return load_attack_stix(str(path), name="attack")


def test_terms_built(graph):
    assert graph.name == "attack"
    assert graph.version == "2.1"
    # tactic + 3 attack-patterns (including the revoked one as a term)
    assert "execution" in graph.terms
    assert "T1059" in graph.terms
    assert "T1059.001" in graph.terms
    assert "T9999" in graph.terms
    assert graph.terms["T1059"].name == "Command and Scripting Interpreter"
    assert graph.terms["execution"].name == "Execution"


def test_subtechnique_is_a_edge(graph):
    assert "T1059" in graph.terms["T1059.001"].parents
    edges = set(graph.is_a_edges())
    assert ("T1059.001", "T1059") in edges


def test_kill_chain_typed_relation(graph):
    rels = set(graph.typed_relations())
    assert ("T1059", "accomplishes", "execution") in rels
    assert ("T1059.001", "accomplishes", "execution") in rels


def test_uses_relation(graph):
    rels = set(graph.typed_relations())
    assert ("T1059", "uses", "T1059.001") in rels


def test_revoked_is_obsolete(graph):
    assert graph.terms["T9999"].obsolete is True
    # obsolete terms are excluded from the active view + edge iterators
    assert "T9999" not in graph.active_terms()


def test_mounts_via_shared_path(graph):
    """The synthetic ATT&CK graph mounts through the same pipeline as HPO."""
    mounted = mount(graph)
    ancestors = mounted.canon.lattice.ancestors(fid("T1059.001"))
    assert fid("T1059") in ancestors
    case = mounted.build_case(["T1059", "T1059.001"])
    functors = case.functor_counts()
    assert functors.get(fid("T1059")) == 1
    assert functors.get(fid("T1059.001")) == 1
    # uses(T1059, T1059.001) higher-order statement present (both endpoints in)
    assert functors.get("uses") == 1
