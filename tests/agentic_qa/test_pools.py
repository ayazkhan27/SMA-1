"""Tests for sma.eval.agentic_qa.pools: registered LLM-QA question pools.

Uses a tiny synthetic mounted ontology + a tiny ``phenotype.hpoa`` temp file (no
real HPO download), and checks the pre-registration contract:
``index_items`` non-empty, answerable golds ARE indexed, ook/novel golds are NOT,
``QAItem`` fields populated, and pools deterministic under a fixed seed.
"""

from __future__ import annotations

import pathlib

import pytest

from sma.eval.agentic_qa.pools import QAItem, build_pools
from sma.ontology.graph import OntologyGraph, Term
from sma.ontology.mount import mount

# A tiny HPO-like ontology: a 2-level is_a tree so the query generator can climb.
#   HP:0000001 (root) -> {HP:0000100 abnormalA, HP:0000200 abnormalB}
#   abnormalA -> {HP:0001001 ph1, HP:0001002 ph2, HP:0001003 ph3}
#   abnormalB -> {HP:0002001 ph4, HP:0002002 ph5, HP:0002003 ph6}
_TERMS = {
    "HP:0000001": Term(id="HP:0000001", name="root"),
    "HP:0000100": Term(id="HP:0000100", name="abnormal A", parents=("HP:0000001",)),
    "HP:0000200": Term(id="HP:0000200", name="abnormal B", parents=("HP:0000001",)),
    "HP:0001001": Term(id="HP:0001001", name="phenotype one", parents=("HP:0000100",)),
    "HP:0001002": Term(id="HP:0001002", name="phenotype two", parents=("HP:0000100",)),
    "HP:0001003": Term(id="HP:0001003", name="phenotype three", parents=("HP:0000100",)),
    "HP:0002001": Term(id="HP:0002001", name="phenotype four", parents=("HP:0000200",)),
    "HP:0002002": Term(id="HP:0002002", name="phenotype five", parents=("HP:0000200",)),
    "HP:0002003": Term(id="HP:0002003", name="phenotype six", parents=("HP:0000200",)),
}

# Six diseases, each with 4 leaf phenotypes (in [min_ph=3, max_ph=8]). One row
# carries an unknown term + a non-"P" aspect row to exercise the hpoa filters.
_DISEASES = {
    "OMIM:1": ("Disease One", ["HP:0001001", "HP:0001002", "HP:0001003", "HP:0002001"]),
    "OMIM:2": ("Disease Two", ["HP:0001001", "HP:0001002", "HP:0002001", "HP:0002002"]),
    "OMIM:3": ("Disease Three", ["HP:0001002", "HP:0001003", "HP:0002002", "HP:0002003"]),
    "OMIM:4": ("Disease Four", ["HP:0001001", "HP:0001003", "HP:0002001", "HP:0002003"]),
    "OMIM:5": ("Disease Five", ["HP:0001001", "HP:0002001", "HP:0002002", "HP:0002003"]),
    "OMIM:6": ("Disease Six", ["HP:0001002", "HP:0002001", "HP:0002002", "HP:0001003"]),
}

# hpoa columns: 0=id 1=name 2=ref 3=hpo_term ... 10=aspect (P kept). 11 cols total.
_NCOLS = 11
_ASPECT_COL = 10


def _hpoa_row(disease_id: str, name: str, hpo_term: str, aspect: str = "P") -> str:
    cols = [""] * _NCOLS
    cols[0] = disease_id
    cols[1] = name
    cols[3] = hpo_term
    cols[_ASPECT_COL] = aspect
    return "\t".join(cols)


@pytest.fixture
def mounted():
    return mount(OntologyGraph(name="hpo", terms=dict(_TERMS)))


@pytest.fixture
def hpoa_path(tmp_path: pathlib.Path) -> str:
    lines = [
        "#description: synthetic",
        "database_id\tdisease_name\t...",  # header line that must be skipped
    ]
    for did, (name, terms) in _DISEASES.items():
        for t in terms:
            lines.append(_hpoa_row(did, name, t))
    # noise the filters must drop: a non-phenotype aspect row + an unknown term.
    lines.append(_hpoa_row("OMIM:1", "Disease One", "HP:0009999", aspect="C"))
    lines.append(_hpoa_row("OMIM:1", "Disease One", "HP:0008888", aspect="P"))  # unknown term, kept-aspect
    path = tmp_path / "phenotype.hpoa"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _build(mounted, hpoa_path, **kw):
    params = dict(seed=7, n_index=4, n_answerable=3, n_held=2, min_ph=3, max_ph=8)
    params.update(kw)
    return build_pools(mounted, hpoa_path, **params)


def test_pools_structure_and_membership(mounted, hpoa_path):
    pools = _build(mounted, hpoa_path)
    assert set(pools) == {
        "index_items",
        "answerable",
        "ook",
        "novel",
        "calib_answerable",
        "calib_ook",
    }

    index_items = pools["index_items"]
    assert index_items, "index_items must be non-empty"
    assert len(index_items) == 4  # n_index of the 6 eligible diseases
    indexed_keys = {it.key for it in index_items}

    # answerable golds ARE in the index; ook/novel golds are NOT.
    assert pools["answerable"], "answerable pool must be non-empty"
    assert pools["ook"], "ook/novel pool must be non-empty"
    for qa in pools["answerable"]:
        assert qa.answerable is True and qa.novel is False
        assert qa.gold_id in indexed_keys
    for qa in pools["ook"]:
        assert qa.answerable is False and qa.novel is True
        assert qa.gold_id not in indexed_keys

    # ook and novel are the same list (held-out == unanswerable AND novel).
    assert pools["ook"] is pools["novel"]


def test_pool_sizes_respect_budgets(mounted, hpoa_path):
    pools = _build(mounted, hpoa_path)  # n_answerable=3, n_held=2
    assert len(pools["answerable"]) == 3
    assert len(pools["novel"]) == 2  # only 2 diseases held out (6 - n_index=4)


def test_qaitem_fields_populated(mounted, hpoa_path):
    pools = _build(mounted, hpoa_path)
    for qa in pools["answerable"] + pools["novel"]:
        assert isinstance(qa, QAItem)
        assert qa.case_text.startswith("Patient presents with: ")
        assert qa.case_terms  # non-empty frozenset
        assert isinstance(qa.case_terms, frozenset)
        assert qa.gold_id and qa.gold_name
        # gold_name is the parsed disease name, not the id.
        assert qa.gold_name.startswith("Disease ")


def test_index_items_carry_known_term_names(mounted, hpoa_path):
    pools = _build(mounted, hpoa_path)
    for it in pools["index_items"]:
        # text is space-joined term NAMES (e.g. "phenotype one"), not raw ids.
        assert "HP:" not in it.text
        assert it.text
        # term_ids are all known and within the eligibility band.
        assert it.term_ids
        assert all(t in mounted.graph.terms for t in it.term_ids)
        assert it.meta["name"].startswith("Disease ")


def test_hpoa_filters_drop_unknown_and_nonphenotype(mounted, hpoa_path):
    pools = _build(mounted, hpoa_path)
    # The unknown HP:0008888 and the aspect="C" HP:0009999 must never appear in
    # any indexed term-set (only known, aspect-P phenotypes survive).
    all_indexed_terms = {t for it in pools["index_items"] for t in it.term_ids}
    assert "HP:0008888" not in all_indexed_terms
    assert "HP:0009999" not in all_indexed_terms


def test_deterministic_under_seed(mounted, hpoa_path):
    a = _build(mounted, hpoa_path)
    b = _build(mounted, hpoa_path)
    # Same selection (ids) AND same generated cases (text + terms).
    assert [it.key for it in a["index_items"]] == [it.key for it in b["index_items"]]
    assert [(q.gold_id, q.case_text, q.case_terms) for q in a["answerable"]] == [
        (q.gold_id, q.case_text, q.case_terms) for q in b["answerable"]
    ]
    assert [(q.gold_id, q.case_text, q.case_terms) for q in a["novel"]] == [
        (q.gold_id, q.case_text, q.case_terms) for q in b["novel"]
    ]


def test_different_seed_changes_split_or_cases(mounted, hpoa_path):
    a = _build(mounted, hpoa_path, seed=7)
    b = _build(mounted, hpoa_path, seed=99)
    a_sig = ([it.key for it in a["index_items"]], [q.case_text for q in a["answerable"]])
    b_sig = ([it.key for it in b["index_items"]], [q.case_text for q in b["answerable"]])
    assert a_sig != b_sig


def test_calibration_pools_disjoint_and_labelled(mounted, hpoa_path):
    # Leave spares on both sides: 4 indexed (2 test + spares), 2 held (1 test + 1 spare).
    pools = _build(mounted, hpoa_path, n_index=4, n_answerable=2, n_held=1, n_calib=1)
    calib = pools["calib_answerable"] + pools["calib_ook"]
    assert calib, "calibration pools must be non-empty when spare diseases remain"

    # Disjoint from the test split (no leakage of test cases into calibration).
    test_ids = {q.gold_id for q in pools["answerable"]} | {q.gold_id for q in pools["novel"]}
    assert all(q.gold_id not in test_ids for q in calib)

    # calib_answerable golds ARE indexed (should-answer); calib_ook are NOT.
    indexed = {it.key for it in pools["index_items"]}
    for q in pools["calib_answerable"]:
        assert q.answerable is True and q.gold_id in indexed
    for q in pools["calib_ook"]:
        assert q.answerable is False and q.gold_id not in indexed


def test_calibration_pools_do_not_perturb_test_cases(mounted, hpoa_path):
    # Adding calibration draws must not change the test pools (drawn first).
    a = _build(mounted, hpoa_path, n_calib=0)
    b = _build(mounted, hpoa_path, n_calib=1)
    assert [(q.gold_id, q.case_text) for q in a["answerable"]] == [
        (q.gold_id, q.case_text) for q in b["answerable"]
    ]
    assert [(q.gold_id, q.case_text) for q in a["novel"]] == [
        (q.gold_id, q.case_text) for q in b["novel"]
    ]
