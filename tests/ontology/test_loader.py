"""Tests for the OBO/OWL ontology loaders."""

from __future__ import annotations

from pathlib import Path

import pytest

from sma.ontology.graph import OntologyGraph
from sma.ontology.loader import fid, load_obo, load_owl, load_ontology

REAL_HPO = Path("/mnt/zephyr27/Documents/SMA-1/data/raw/hpo/hp.obo")

OBO_FIXTURE = """\
format-version: 1.2
data-version: tiny/releases/2026-01-01
ontology: tiny

[Term]
id: TST:0000001
name: Root

[Term]
id: TST:0000002
name: Child
is_a: TST:0000001 ! Root
relationship: part_of TST:0000001 ! Root

[Term]
id: TST:0000003
name: Old term
is_a: TST:0000001 ! Root
is_obsolete: true

[Typedef]
id: part_of
name: part of
"""

OWL_FIXTURE = """\
<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
         xmlns:owl="http://www.w3.org/2002/07/owl#">
  <owl:Ontology rdf:about="http://example.org/tiny">
    <owl:versionInfo>2026-01-01</owl:versionInfo>
  </owl:Ontology>
  <owl:Class rdf:about="http://purl.obolibrary.org/obo/TST_0000001">
    <rdfs:label>Root</rdfs:label>
  </owl:Class>
  <owl:Class rdf:about="http://purl.obolibrary.org/obo/TST_0000002">
    <rdfs:label>Child</rdfs:label>
    <rdfs:subClassOf rdf:resource="http://purl.obolibrary.org/obo/TST_0000001"/>
    <rdfs:subClassOf>
      <owl:Restriction>
        <owl:onProperty rdf:resource="http://purl.obolibrary.org/obo/part_of"/>
        <owl:someValuesFrom rdf:resource="http://purl.obolibrary.org/obo/TST_0000001"/>
      </owl:Restriction>
    </rdfs:subClassOf>
  </owl:Class>
</rdf:RDF>
"""


def test_load_obo_parses_terms_edges_and_skips_obsolete(tmp_path: Path) -> None:
    obo = tmp_path / "tiny.obo"
    obo.write_text(OBO_FIXTURE, encoding="utf-8")

    g = load_obo(str(obo), name="")
    assert isinstance(g, OntologyGraph)
    assert g.name == "tiny"
    assert g.version == "tiny/releases/2026-01-01"

    assert set(g.terms) == {"TST:0000001", "TST:0000002", "TST:0000003"}
    child = g.terms["TST:0000002"]
    assert child.name == "Child"
    assert child.parents == ("TST:0000001",)
    assert child.relations == (("part_of", "TST:0000001"),)
    assert g.terms["TST:0000003"].obsolete is True

    # Obsolete term excluded from active set.
    assert set(g.active_terms()) == {"TST:0000001", "TST:0000002"}

    # is_a edges skip the obsolete child's edge.
    edges = sorted(g.is_a_edges())
    assert edges == [("TST:0000002", "TST:0000001")]

    rels = sorted(g.typed_relations())
    assert rels == [("TST:0000002", "part_of", "TST:0000001")]


def test_load_owl_recovers_edges_relation_and_label(tmp_path: Path) -> None:
    owl = tmp_path / "tiny.owl"
    owl.write_text(OWL_FIXTURE, encoding="utf-8")

    g = load_owl(str(owl), name="tiny")
    assert g.version == "2026-01-01"
    assert set(g.terms) == {"TST:0000001", "TST:0000002"}
    assert g.terms["TST:0000001"].name == "Root"

    child = g.terms["TST:0000002"]
    assert child.name == "Child"
    assert child.parents == ("TST:0000001",)
    assert child.relations == (("part_of", "TST:0000001"),)

    assert sorted(g.is_a_edges()) == [("TST:0000002", "TST:0000001")]
    assert sorted(g.typed_relations()) == [("TST:0000002", "part_of", "TST:0000001")]


def test_load_ontology_dispatch(tmp_path: Path) -> None:
    obo = tmp_path / "tiny.obo"
    obo.write_text(OBO_FIXTURE, encoding="utf-8")
    owl = tmp_path / "tiny.owl"
    owl.write_text(OWL_FIXTURE, encoding="utf-8")

    assert load_ontology(str(obo)).name == "tiny"
    assert set(load_ontology(str(owl)).terms) == {"TST:0000001", "TST:0000002"}


def test_fid() -> None:
    assert fid("HP:0001250") == "HP_0001250"


def test_real_hpo_smoke() -> None:
    if not REAL_HPO.exists():
        pytest.skip("real HPO data not available")
    g = load_obo(str(REAL_HPO), name="hpo")
    assert len(g.active_terms()) > 15000
    assert sum(1 for _ in g.is_a_edges()) > 20000
    assert g.version
