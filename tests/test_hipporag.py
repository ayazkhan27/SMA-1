from __future__ import annotations

from sma.eval.baselines.hipporag import (
    HippoRAGRetriever,
    _line_triples,
    extract_phrases,
    rank_hipporag,
)

DOCS = [
    (
        "doc_timeout",
        "ERROR db.ConnectionPool: DatabaseTimeoutException raised\n"
        "db connection timeout while sending query to 10.0.0.5:5432\n"
        "retrying connection to db host db-primary failed",
    ),
    (
        "doc_block",
        "INFO dfs.DataNode$PacketResponder: Received block blk_-160899 of size 67108864 from 10.251.42.84\n"
        "INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated",
    ),
    (
        "doc_cron",
        "INFO scheduler started cron job for cleanup task\n"
        "cleanup task completed in 42ms",
    ),
]


def test_extraction_finds_entities_and_relations():
    triples, pairs, entities = _line_triples(
        "ERROR db.ConnectionPool: DatabaseTimeoutException raised by worker-3"
    )
    assert "db.ConnectionPool" in entities
    assert "DatabaseTimeoutException" in entities
    assert "worker-3" in entities
    assert any(rel == "raised" for _, rel, _ in triples)
    assert pairs == []  # relation present -> triples, not co-occurrence

    phrases = extract_phrases("db connection timeout for blk_-160899 at 10.0.0.5")
    assert {"db", "connection", "timeout", "blk_-160899", "10.0.0.5"} <= set(phrases)


def test_determinism_same_input_identical_ranking():
    first = rank_hipporag("timeout db connection", DOCS, k=3)
    second = rank_hipporag("timeout db connection", DOCS, k=3)
    assert first == second

    retriever = HippoRAGRetriever()
    retriever.build(DOCS)
    a = retriever.retrieve("Received block blk_-160899", k=3)
    b = retriever.retrieve("Received block blk_-160899", k=3)
    assert a == b
    assert [doc_id for doc_id, _ in a][0] == "doc_block"


def test_timeout_query_ranks_timeout_doc_first():
    ranked = rank_hipporag("timeout db", DOCS, k=3)
    ids = [doc_id for doc_id, _ in ranked]
    assert ids[0] == "doc_timeout"
    assert ranked[0][1] > dict(ranked)["doc_cron"]


def test_synonym_edge_links_normalized_variants():
    docs = DOCS + [("doc_syn", "WARN Database.Timeout.Exception observed on replica")]
    retriever = HippoRAGRetriever()
    retriever.build(docs)
    g = retriever.graph
    assert g.has_edge("e::DatabaseTimeoutException", "e::Database.Timeout.Exception")


def test_empty_index_returns_empty():
    retriever = HippoRAGRetriever()
    retriever.build([])
    assert retriever.retrieve("anything", k=5) == []
