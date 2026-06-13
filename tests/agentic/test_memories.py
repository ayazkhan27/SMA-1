"""Tests for the agentic Memory wrappers (SMA + enterprise-RAG gauntlet).

Each memory indexes ~3 toy docs and must rank an exact-match query's key at
rank 1 with confidence in [0,1]. BGE-embedder and cross-encoder reranker tests
load real models on CPU and are marked ``@pytest.mark.slow``.
"""

from __future__ import annotations

import pytest

from sma.eval.agentic.memories import (
    BM25Memory,
    DenseMemory,
    HippoMemory,
    HybridRerankMemory,
    HybridRRFMemory,
    IndexItem,
    Query,
    Retrieved,
    SmaMemory,
)
from sma.ontology.graph import OntologyGraph, Term
from sma.ontology.mount import mount


# ---------------------------------------------------------------------------
# Fixtures: a small is-a ontology + three toy entities.
# ---------------------------------------------------------------------------


@pytest.fixture
def mounted():
    """Six-term is-a ontology: leaves A,B,C,D under parents P,Q."""
    graph = OntologyGraph(
        name="toy",
        version="v0",
        terms={
            "TST:A": Term(id="TST:A", name="alpha cardiac murmur", parents=("TST:P",)),
            "TST:B": Term(id="TST:B", name="beta retinal pigment", parents=("TST:P",)),
            "TST:C": Term(id="TST:C", name="gamma hepatic fibrosis", parents=("TST:Q",)),
            "TST:D": Term(id="TST:D", name="delta renal cyst", parents=("TST:Q",)),
            "TST:P": Term(id="TST:P", name="parent organ system one"),
            "TST:Q": Term(id="TST:Q", name="parent organ system two"),
        },
    )
    return mount(graph)


def _names(mounted, ids):
    return " ".join(mounted.graph.terms[t].name for t in ids)


@pytest.fixture
def items(mounted):
    """Three entities with disjoint term sets so an exact query is unambiguous."""
    specs = {
        "ent_ab": ["TST:A", "TST:B"],
        "ent_cd": ["TST:C", "TST:D"],
        "ent_ac": ["TST:A", "TST:C"],
    }
    return [
        IndexItem(key=key, term_ids=frozenset(ids), text=_names(mounted, ids))
        for key, ids in specs.items()
    ]


def _query(mounted, ids):
    return Query(term_ids=frozenset(ids), text=_names(mounted, ids))


def _assert_top1(results, expected_key):
    assert results, "memory returned no results"
    # monotonic ranks 1..k
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    for r in results:
        assert isinstance(r, Retrieved)
        assert 0.0 <= r.confidence <= 1.0
    assert results[0].key == expected_key


# ---------------------------------------------------------------------------
# Task 1: protocol contract on a trivial stub.
# ---------------------------------------------------------------------------


def test_stub_memory_monotonic_ranks():
    class StubMemory:
        name = "stub"

        def index(self, items):
            self._keys = [it.key for it in items]

        def retrieve(self, query, k):
            return [
                Retrieved(key, score=float(len(self._keys) - i), confidence=0.5, rank=i + 1)
                for i, key in enumerate(self._keys[:k])
            ]

        def novelty(self, query):
            return 0.0

    mem = StubMemory()
    mem.index([IndexItem(key="x", term_ids=frozenset(), text="x")])
    res = mem.retrieve(Query(frozenset(), "x"), k=5)
    assert [r.rank for r in res] == list(range(1, len(res) + 1))
    assert all(0.0 <= r.confidence <= 1.0 for r in res)
    assert 0.0 <= mem.novelty(Query(frozenset(), "x")) <= 1.0


# ---------------------------------------------------------------------------
# Task 2: SMA memory.
# ---------------------------------------------------------------------------


def test_sma_exact_match_rank1(mounted, items):
    mem = SmaMemory(mounted)
    mem.index(items)
    res = mem.retrieve(_query(mounted, ["TST:A", "TST:B"]), k=3)
    _assert_top1(res, "ent_ab")
    assert res[0].confidence > 0.0


def test_sma_novelty_higher_for_unrelated(mounted, items):
    mem = SmaMemory(mounted)
    mem.index(items)
    in_dist = mem.novelty(_query(mounted, ["TST:A", "TST:B"]))
    out_dist = mem.novelty(_query(mounted, ["TST:D"]))
    assert 0.0 <= in_dist <= 1.0
    assert 0.0 <= out_dist <= 1.0
    assert out_dist >= in_dist


# ---------------------------------------------------------------------------
# Task 3: BM25, Dense (BGE, slow), HippoRAG.
# ---------------------------------------------------------------------------


def test_bm25_exact_match_rank1(mounted, items):
    mem = BM25Memory()
    mem.index(items)
    res = mem.retrieve(_query(mounted, ["TST:A", "TST:B"]), k=3)
    _assert_top1(res, "ent_ab")
    assert 0.0 <= mem.novelty(_query(mounted, ["TST:A", "TST:B"])) <= 1.0


def test_hipporag_exact_match_rank1(mounted, items):
    mem = HippoMemory()
    mem.index(items)
    res = mem.retrieve(_query(mounted, ["TST:A", "TST:B"]), k=3)
    _assert_top1(res, "ent_ab")
    assert 0.0 <= mem.novelty(_query(mounted, ["TST:A", "TST:B"])) <= 1.0


@pytest.mark.slow
def test_dense_exact_match_rank1(mounted, items):
    mem = DenseMemory()
    mem.index(items)
    res = mem.retrieve(_query(mounted, ["TST:A", "TST:B"]), k=3)
    _assert_top1(res, "ent_ab")
    assert 0.0 <= mem.novelty(_query(mounted, ["TST:A", "TST:B"])) <= 1.0


# ---------------------------------------------------------------------------
# Task 4: Hybrid-RRF + Hybrid+Rerank.
# ---------------------------------------------------------------------------


def test_hybrid_rrf_agreement_rank1(mounted):
    """RRF ranks first the doc both members agree on."""

    class FakeMem:
        def __init__(self, name, order):
            self.name = name
            self._order = order

        def index(self, items):
            self._items = items

        def retrieve(self, query, k):
            return [
                Retrieved(key, score=float(len(self._order) - i), confidence=0.5, rank=i + 1)
                for i, key in enumerate(self._order[:k])
            ]

        def novelty(self, query):
            return 0.3

    items = [IndexItem(key=k, term_ids=frozenset(), text=k) for k in ("a", "b", "c")]
    # both members rank "b" highly: bm25 -> b,a,c ; dense -> b,c,a
    bm25 = FakeMem("bm25", ["b", "a", "c"])
    dense = FakeMem("dense", ["b", "c", "a"])
    mem = HybridRRFMemory(bm25, dense)
    mem.index(items)
    res = mem.retrieve(Query(frozenset(), ""), k=3)
    _assert_top1(res, "b")
    assert mem.novelty(Query(frozenset(), "")) == 0.3


@pytest.mark.slow
def test_hybrid_rerank_reorders(mounted, items):
    """Cross-encoder reranks the hybrid candidates by (query, text) relevance."""
    bm25, dense = BM25Memory(), DenseMemory()
    hybrid = HybridRRFMemory(bm25, dense)
    mem = HybridRerankMemory(hybrid, top_n=3)
    mem.index(items)
    res = mem.retrieve(_query(mounted, ["TST:A", "TST:B"]), k=3)
    _assert_top1(res, "ent_ab")
