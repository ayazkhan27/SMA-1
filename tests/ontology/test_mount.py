"""Tests for sma.ontology.mount: lattice mounting, case + index building."""

from __future__ import annotations

import pytest

from sma.ontology.graph import OntologyGraph, Term
from sma.ontology.loader import fid
from sma.ontology.mount import MountedOntology, mount


@pytest.fixture
def toy_graph() -> OntologyGraph:
    """A 3-term toy ontology: A is_a B, and A part_of C."""
    return OntologyGraph(
        name="toy",
        version="v0",
        terms={
            "TST:A": Term(id="TST:A", name="a", parents=("TST:B",),
                          relations=(("part_of", "TST:C"),)),
            "TST:B": Term(id="TST:B", name="b"),
            "TST:C": Term(id="TST:C", name="c"),
        },
    )


def test_mount_lattice_has_is_a_edge(toy_graph: OntologyGraph) -> None:
    mounted = mount(toy_graph)
    assert isinstance(mounted, MountedOntology)
    ancestors = mounted.canon.lattice.ancestors(fid("TST:A"))
    assert fid("TST:B") in ancestors
    # default config: ascend two is_a hops with rho=0.95
    assert mounted.config.delta == 2
    assert mounted.config.rho == 0.95


def test_build_case_includes_higher_order_when_both_endpoints_present(toy_graph):
    mounted = mount(toy_graph)
    case = mounted.build_case(["TST:A", "TST:C"])
    functors = case.functor_counts()
    # two unary term statements
    assert functors.get(fid("TST:A")) == 1
    assert functors.get(fid("TST:C")) == 1
    # the higher-order relation is present
    assert functors.get("part_of") == 1
    # top-level statements: A(subj), C(subj), part_of(A(subj), C(subj))
    assert len(case.statements) == 3


def test_build_case_omits_relation_when_only_one_endpoint_present(toy_graph):
    mounted = mount(toy_graph)
    case = mounted.build_case(["TST:A"])  # C absent -> no part_of
    functors = case.functor_counts()
    assert functors.get(fid("TST:A")) == 1
    assert "part_of" not in functors
    assert len(case.statements) == 1


def test_build_case_drops_unknown_terms(toy_graph):
    mounted = mount(toy_graph)
    case = mounted.build_case(["TST:A", "TST:UNKNOWN"])
    functors = case.functor_counts()
    assert functors.get(fid("TST:A")) == 1
    assert fid("TST:UNKNOWN") not in functors


def test_build_index_returns_recoverable_keys(toy_graph):
    mounted = mount(toy_graph)
    records = [
        ("rec_ac", ["TST:A", "TST:C"], None),
        ("rec_b", ["TST:B"], {"extra": 1}),
    ]
    index = mounted.build_index(records)

    # key_of maps every indexed case_id back to its record key
    assert hasattr(index, "key_of")
    assert set(index.key_of.values()) == {"rec_ac", "rec_b"}
    # metadata["key"] is stamped on each case
    for case in index.cases.values():
        assert index.key_of[case.case_id] == case.metadata["key"]

    # retrieving with a query matching the first record recovers its key
    query = mounted.build_case(["TST:A", "TST:C"])
    results = index.retrieve(query, k=2, shortlist=10, fac_budget=10)
    assert results
    keys = [index.key_of[r.case_id] for r in results]
    assert "rec_ac" in keys
    # the exact-shape record is the top hit
    assert index.key_of[results[0].case_id] == "rec_ac"
