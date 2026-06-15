"""Tests for the SMA-1 MCP server engine (sma.mcp).

The engine is exercised without the `mcp` transport SDK; a separate test builds
the FastMCP server only when `mcp` is importable.
"""
from __future__ import annotations

import pytest

from sma.mcp import SUPPORTED_FORMATS, SmaEngine, load_manifest

TOY_OBO = """format-version: 1.2

[Term]
id: T:001
name: root

[Term]
id: T:002
name: alpha mechanism
is_a: T:001

[Term]
id: T:003
name: beta mechanism
is_a: T:001

[Term]
id: T:004
name: gamma variant
is_a: T:002
"""


@pytest.fixture()
def obo(tmp_path):
    p = tmp_path / "toy.obo"
    p.write_text(TOY_OBO)
    return str(p)


@pytest.fixture()
def engine(obo):
    e = SmaEngine()
    e.mount_ontology("toy", obo, "obo")
    e.index_cases(
        "toy",
        [
            {"key": "CASE_A", "term_ids": ["T:002"], "text": "alpha mechanism"},
            {"key": "CASE_B", "term_ids": ["T:003"], "text": "beta mechanism"},
            {"key": "CASE_C", "term_ids": ["T:004"], "text": "gamma variant"},
        ],
    )
    return e


def test_mount_reports_concepts(engine):
    onts = engine.list_ontologies()
    toy = next(o for o in onts["ontologies"] if o["name"] == "toy")
    assert toy["mounted"] is True
    assert toy["concepts"] == 4
    assert toy["indexed_cases"] == 3
    assert "obo" in onts["supported_formats"]
    assert onts["supported_formats"] == SUPPORTED_FORMATS


def test_encode_text_matches_term_name(engine):
    enc = engine.encode_text("toy", "a study of the alpha mechanism in cells")
    assert "T:002" in enc["term_ids"]
    assert "alpha mechanism" in enc["matched_names"]


def test_retrieve_returns_structural_match(engine):
    r = engine.retrieve("toy", term_ids=["T:002"], k=3)
    assert r["abstain"] is False
    ids = [c["id"] for c in r["citations"]]
    assert "CASE_A" in ids  # the exact-structure case is retrieved
    # every citation carries a checkable id, a raw score, and a normalized confidence
    for c in r["citations"]:
        assert c["id"] and isinstance(c["score"], float)
        assert 0.0 <= c["confidence"] <= 1.0


def test_retrieve_via_text_encodes_then_matches(engine):
    r = engine.retrieve("toy", text="alpha mechanism", k=3)
    assert r["query_term_ids"] == ["T:002"]
    assert any(c["id"] == "CASE_A" for c in r["citations"])


def test_high_threshold_forces_abstain(engine):
    r = engine.retrieve("toy", term_ids=["T:002"], k=3, ground_threshold=1e6)
    assert r["abstain"] is True
    assert r["citations"] == []
    assert "no structural precedent" in (r["note"] or "")


def test_retrieve_unmatched_text_abstains(engine):
    r = engine.retrieve("toy", text="something with no matching term at all", k=3)
    assert r["abstain"] is True
    assert r["query_term_ids"] == []


def test_novelty_returns_bounded_float_and_flag(engine):
    n = engine.novelty("toy", term_ids=["T:002"])
    assert 0.0 <= n["novelty"] <= 1.0
    assert isinstance(n["novelty_flag"], bool)


def test_unregistered_ontology_raises(engine):
    with pytest.raises(KeyError):
        engine.retrieve("does-not-exist", term_ids=["T:002"])


def test_load_manifest_registers_and_indexes(obo, tmp_path):
    import json

    manifest = tmp_path / "m.json"
    manifest.write_text(
        json.dumps(
            {
                "ontologies": [{"name": "toy", "path": obo, "format": "obo"}],
                "cases": {"toy": [{"key": "K1", "term_ids": ["T:002"], "text": "alpha mechanism"}]},
            }
        )
    )
    e = SmaEngine()
    load_manifest(e, str(manifest))
    r = e.retrieve("toy", term_ids=["T:002"], k=1)
    assert any(c["id"] == "K1" for c in r["citations"])


def test_retrieve_with_no_indexed_cases_abstains(obo):
    e = SmaEngine()
    e.mount_ontology("toy", obo, "obo")  # mounted but NO cases indexed
    r = e.retrieve("toy", term_ids=["T:002"], k=3)
    assert r["abstain"] is True
    assert r["citations"] == []
    assert r["reason"] == "no cases indexed"
    assert r["novelty"] is None          # cannot be scored (no index) != "novel"
    assert r["novelty_flag"] is False


def test_encode_text_no_matches_returns_empty(engine):
    enc = engine.encode_text("toy", "this text contains no ontology terms at all")
    assert enc["term_ids"] == []
    assert enc["matched_names"] == []


def test_encode_text_respects_max_terms(engine):
    enc = engine.encode_text("toy", "alpha mechanism and beta mechanism", max_terms=1)
    assert len(enc["term_ids"]) == 1  # longest/most-specific name kept after sort


def test_encode_text_matches_across_punctuation(engine):
    # token-boundary (regex) match: name adjacent to punctuation still matches
    enc = engine.encode_text("toy", "findings: alpha mechanism, confirmed.")
    assert "T:002" in enc["term_ids"]


def test_novelty_is_none_when_unscoreable(obo):
    e = SmaEngine()
    e.mount_ontology("toy", obo, "obo")  # no cases indexed -> novelty unscoreable
    n = e.novelty("toy", term_ids=["T:002"])
    assert n["novelty"] is None
    assert n["novelty_flag"] is False


# -- catalog of public ontologies ------------------------------------------

def test_catalog_browse_and_search():
    e = SmaEngine()
    lc = e.list_catalog()
    assert lc["count"] >= 100            # the bundled catalog ships hundreds of ontologies
    assert len(lc["categories"]) >= 5
    assert e.search_catalog("phenotype")["count"] >= 1
    assert any(o["id"] == "chebi" for o in e.search_catalog("chebi")["ontologies"])


def test_list_catalog_filters_by_category():
    e = SmaEngine()
    cat = e.list_catalog()["categories"][0]
    filtered = e.list_catalog(category=cat)
    assert filtered["count"] >= 1
    assert all(cat.lower() in o["category"].lower() for o in filtered["ontologies"])


def test_mount_from_catalog_offline(obo, tmp_path, monkeypatch):
    # exercise catalog -> fetch -> mount end to end via a file:// URL (no network)
    monkeypatch.setenv("SMA_CACHE_DIR", str(tmp_path / "cache"))
    e = SmaEngine()
    monkeypatch.setattr(
        e, "load_catalog",
        lambda: [{"id": "toy", "name": "Toy", "category": "Test",
                  "format": "obo", "url": f"file://{obo}", "license": "CC0"}],
    )
    r = e.mount_from_catalog("toy")
    assert r["ontology"] == "toy"
    assert r["concepts"] == 4
    e.index_cases("toy", [{"key": "C1", "term_ids": ["T:002"], "text": "alpha"}])
    assert any(c["id"] == "C1" for c in e.retrieve("toy", term_ids=["T:002"], k=1)["citations"])


def test_mount_from_catalog_unknown_id_raises():
    e = SmaEngine()
    with pytest.raises(KeyError):
        e.mount_from_catalog("definitely-not-an-ontology-id")


def test_build_server_registers_tools():
    pytest.importorskip("mcp")
    import asyncio

    from sma.mcp import build_server

    server, engine = build_server()
    assert engine is not None and server is not None
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert {
        "list_ontologies", "mount_ontology", "index_cases", "encode_text",
        "retrieve", "novelty", "list_catalog", "search_catalog", "mount_from_catalog",
    } <= names
