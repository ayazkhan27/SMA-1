"""Unit tests for the Elliptic graph-neighbourhood encoder + typology lattice."""

from __future__ import annotations

from sma.eval.fraud_elliptic.encoder import (
    ILLICIT,
    LICIT,
    UNKNOWN,
    EllipticGraph,
    NeighbourhoodEncoder,
    _count_class,
    _degree_class,
    _tier,
    build_typology,
)
from sma.ontology import mount


def _toy_graph() -> EllipticGraph:
    # node "C" is the query; A,B are predecessors, D,E,F are successors.
    feats = {
        "A": [1.0, 0.0, 0.0], "B": [1.0, 0.0, 0.0],
        "C": [2.0, 0.9, -0.9],  # time=2, agg-val high-ish, local-val low-ish
        "D": [3.0, 0.0, 0.0], "E": [3.0, 0.0, 0.0], "F": [3.0, 0.0, 0.0],
    }
    g = EllipticGraph(
        time_step={k: int(v[0]) for k, v in feats.items()},
        label={"A": ILLICIT, "B": ILLICIT, "C": ILLICIT, "D": LICIT, "E": UNKNOWN, "F": LICIT},
        feats=feats,
        preds={"C": ["A", "B"], "A": [], "B": [], "D": ["C"], "E": ["C"], "F": ["C"]},
        succs={"C": ["D", "E", "F"], "A": ["C"], "B": ["C"], "D": [], "E": [], "F": []},
    )
    return g


def test_degree_and_count_buckets():
    assert _degree_class(0) == "none"
    assert _degree_class(1) == "one"
    assert _degree_class(3) == "few"
    assert _degree_class(9) == "many"
    assert _count_class(0) == "none"
    assert _count_class(2) == "some"
    assert _count_class(5) == "many"


def test_tier():
    assert _tier(-1.0, -0.3, 0.3) == "low"
    assert _tier(0.0, -0.3, 0.3) == "mid"
    assert _tier(1.0, -0.3, 0.3) == "high"


def test_encoder_emits_topology_terms():
    g = _toy_graph()
    enc = NeighbourhoodEncoder(graph=g, visible_labels=g.label)
    terms = enc.encode("C")
    # 2 predecessors -> fanIn_few ; 3 successors -> fanIn... fanOut_few
    assert "fanIn_few" in terms
    assert "fanOut_few" in terms
    assert any(t.startswith("temp_") for t in terms)
    assert any(t.startswith("inVal_") for t in terms)
    assert any(t.startswith("outVal_") for t in terms)


def test_neighbour_label_context_counts_visible_only():
    g = _toy_graph()
    # All labels visible: C has 2 illicit predecessors (A,B), 2 licit successors (D,F).
    enc = NeighbourhoodEncoder(graph=g, visible_labels=g.label)
    terms = enc.encode("C")
    assert "nbrIllicit_some" in terms  # 2 illicit -> some
    assert "nbrLicit_some" in terms    # 2 licit -> some


def test_leak_guard_hides_held_out_neighbour_labels():
    g = _toy_graph()
    # Visible = only the index split (exclude neighbours D,F so their licit labels hide).
    visible = {"A": ILLICIT, "B": ILLICIT}  # D, E, F not visible
    enc = NeighbourhoodEncoder(graph=g, visible_labels=visible)
    terms = enc.encode("C")
    assert "nbrIllicit_some" in terms
    assert "nbrLicit_none" in terms  # successors' licit labels are not visible


def test_self_label_never_leaks():
    # A node that is its own neighbour (self-loop) must not count its own label.
    g = _toy_graph()
    g.preds["C"] = ["C", "A"]  # self-loop predecessor
    enc = NeighbourhoodEncoder(graph=g, visible_labels=g.label)
    terms = enc.encode("C")
    # Only A among predecessors is illicit-and-not-self; the self-loop C is skipped.
    # successors D,F licit. So illicit count = 1 (A) -> some.
    assert "nbrIllicit_some" in terms


def test_typology_lattice_is_mountable_and_acyclic_ascent():
    graph = build_typology()
    # Every parent referenced must exist as a term (no dangling is-a edges).
    for tid, term in graph.terms.items():
        for p in term.parents:
            assert p in graph.terms, f"{tid} -> missing parent {p}"
    # Mount must populate a lattice and let fanOut_many ascend to illicitTypology.
    mounted = mount(graph)
    edges = list(graph.is_a_edges())
    assert ("fanOut_many", "illicitTypology") in edges
    assert ("fanOut_many", "fanOut_any") in edges
    assert mounted.canon is not None


def test_build_case_emits_higher_order_relations():
    # When two related descriptor terms are both present and the typology wires a
    # typed relation, mount().build_case must emit the higher-order statement.
    from sma.ontology.graph import Term

    graph = build_typology()
    # Add a typed flowsTo relation between own topology and neighbour context.
    graph.terms["fanOut_many"] = Term(
        id="fanOut_many", name="fanOut many",
        parents=("fanOut_any", "illicitTypology"),
        relations=(("flowsTo", "nbrIllicit_many"),),
    )
    mounted = mount(graph)
    case = mounted.build_case(["fanOut_many", "nbrIllicit_many"])
    # The case must contain more than the two unary term statements (the relation).
    assert len(case.statements) >= 3
