"""Uniform ``Memory`` interface + six retriever wrappers for the agentic suite.

The agentic harness holds everything fixed except the retrieval ``Memory``: SMA
(the universal ontology adapter) vs an enterprise-RAG/KG gauntlet (BM25, BGE
dense, Hybrid-RRF, Hybrid+Rerank, HippoRAG). Every memory implements the same
three-method contract so the harness never sees a retriever's internals:

    index(items)              -> None      # build over IndexItem records
    retrieve(query, k)        -> list[Retrieved]   # ranked, rank 1..k
    novelty(query)            -> float     # in [0,1], higher = "this is new"

Confidence (on each :class:`Retrieved`) drives cite-or-abstain and is squashed
to ``[0,1]`` per method; novelty is the method's best out-of-distribution signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Protocol

from sma.eval.baselines.bm25 import rank_bm25_like
from sma.eval.baselines.hipporag import HippoRAGRetriever
from sma.index.macfac import MacFacIndex
from sma.ontology import MountedOntology
from sma.sage.pools import SagePool


@dataclass
class IndexItem:
    """One indexable entity (the gold answer is its ``key``)."""

    key: str  # entity id (gold answer)
    term_ids: frozenset[str]  # ontology term ids (for SMA)
    text: str  # space-joined term NAMES (for text baselines)
    meta: dict = field(default_factory=dict)


@dataclass
class Query:
    """A retrieval query in both ontology-term and text form."""

    term_ids: frozenset[str]
    text: str


@dataclass
class Retrieved:
    """A single ranked hit returned by a :class:`Memory`."""

    key: str
    score: float
    confidence: float  # drives cite-or-abstain; method-specific, in [0,1]
    rank: int


class Memory(Protocol):
    """The only thing that varies across harness runs."""

    name: str

    def index(self, items: list[IndexItem]) -> None: ...

    def retrieve(self, query: Query, k: int) -> list[Retrieved]: ...

    def novelty(self, query: Query) -> float:  # higher = more "this is new"
        ...


# ---------------------------------------------------------------------------
# Task 2: SMA memory (universal adapter)
# ---------------------------------------------------------------------------


class SmaMemory:
    """SMA: mount an ontology, index cases via MacFac, novelty via SagePool."""

    name = "sma"

    def __init__(self, mounted: MountedOntology):
        self.mounted = mounted

    def index(self, items: list[IndexItem]) -> None:
        self._key: dict[str, str] = {}
        cases = []
        self.pool = SagePool("agentic", assimilation_threshold=0.2)
        for it in items:
            c = self.mounted.build_case(it.term_ids, metadata={"key": it.key})
            self._key[c.case_id] = it.key
            cases.append(c)
            self.pool.assimilate(c)
        self.index_ = MacFacIndex(config=self.mounted.config, canon=self.mounted.canon)
        self.index_.build(cases)

    def retrieve(self, query: Query, k: int) -> list[Retrieved]:
        qc = self.mounted.build_case(query.term_ids)
        res = self.index_.retrieve(qc, k=k, shortlist=80, fac_budget=40)
        if not res:
            return []
        top = max(r.score for r in res) or 1.0
        out = []
        for i, r in enumerate(res, 1):
            conf = min(max(r.score / top, 0.0), 1.0)
            out.append(Retrieved(self._key.get(r.case_id, ""), r.score, conf, i))
        return out

    def novelty(self, query: Query) -> float:
        return self.pool.expectation_violation(self.mounted.build_case(query.term_ids))


# ---------------------------------------------------------------------------
# Task 3: text baselines — BM25, Dense (BGE), HippoRAG
# ---------------------------------------------------------------------------


class BM25Memory:
    """Lexical BM25-like baseline over term-name documents."""

    name = "bm25"

    def index(self, items: list[IndexItem]) -> None:
        self._docs: list[tuple[str, str]] = [(it.key, it.text) for it in items]

    def retrieve(self, query: Query, k: int) -> list[Retrieved]:
        ranked = rank_bm25_like(query.text, self._docs, k=k)
        if not ranked:
            return []
        top = ranked[0][1]
        conf = top / (top + 1.0) if top > 0 else 0.0
        return [Retrieved(key, score, conf, i) for i, (key, score) in enumerate(ranked, 1)]

    def novelty(self, query: Query) -> float:
        ranked = rank_bm25_like(query.text, self._docs, k=1)
        if not ranked:
            return 1.0
        top = ranked[0][1]
        conf = top / (top + 1.0) if top > 0 else 0.0
        return 1.0 - conf


_BGE: dict = {}


def _bge():
    """Load the BGE-small embedder once (cached at module level)."""
    if "m" not in _BGE:
        from sentence_transformers import SentenceTransformer

        _BGE["m"] = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _BGE["m"]


class DenseMemory:
    """Neural dense baseline: BGE-small embeddings + cosine similarity."""

    name = "dense"

    def index(self, items: list[IndexItem]) -> None:
        self._keys = [it.key for it in items]
        texts = [it.text for it in items]
        model = _bge()
        # normalize_embeddings=True -> cosine == dot product
        self._mat = model.encode(texts, normalize_embeddings=True)

    def _scores(self, query: Query):
        model = _bge()
        q = model.encode([query.text], normalize_embeddings=True)[0]
        return [float(sum(q[d] * row[d] for d in range(len(q)))) for row in self._mat]

    def retrieve(self, query: Query, k: int) -> list[Retrieved]:
        if not self._keys:
            return []
        scores = self._scores(query)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], self._keys[i]))[:k]
        top = max(scores) if scores else 0.0
        conf = min(max(top, 0.0), 1.0)
        return [Retrieved(self._keys[i], scores[i], conf, rank) for rank, i in enumerate(order, 1)]

    def novelty(self, query: Query) -> float:
        if not self._keys:
            return 1.0
        top = max(self._scores(query))
        return 1.0 - min(max(top, 0.0), 1.0)


class HippoMemory:
    """HippoRAG-2-style KG retrieval (phrase graph + Personalized PageRank)."""

    name = "hipporag"

    def index(self, items: list[IndexItem]) -> None:
        self._retriever = HippoRAGRetriever()
        self._retriever.build([(it.key, it.text) for it in items])
        self._n = len(items)

    def retrieve(self, query: Query, k: int) -> list[Retrieved]:
        ranked = self._retriever.retrieve(query.text, k=k)
        if not ranked:
            return []
        total = sum(s for _, s in ranked) or 1.0
        conf = min(max(ranked[0][1] / total, 0.0), 1.0)
        return [Retrieved(key, score, conf, i) for i, (key, score) in enumerate(ranked, 1)]

    def novelty(self, query: Query) -> float:
        ranked = self._retriever.retrieve(query.text, k=max(1, self._n))
        if not ranked:
            return 1.0
        total = sum(s for _, s in ranked) or 1.0
        conf = min(max(ranked[0][1] / total, 0.0), 1.0)
        return 1.0 - conf


# ---------------------------------------------------------------------------
# Task 4: SOTA hybrid — Hybrid-RRF + Hybrid+Rerank
# ---------------------------------------------------------------------------


class HybridRRFMemory:
    """Reciprocal Rank Fusion of a BM25 and a dense member memory."""

    name = "hybrid_rrf"

    def __init__(self, bm25_mem: BM25Memory, dense_mem: DenseMemory, k_rrf: int = 60):
        self.bm25_mem = bm25_mem
        self.dense_mem = dense_mem
        self.k_rrf = k_rrf

    def index(self, items: list[IndexItem]) -> None:
        self.bm25_mem.index(items)
        self.dense_mem.index(items)
        self._n = len(items)

    def _fused(self, query: Query, k: int) -> list[tuple[str, float]]:
        depth = max(k, self._n)
        fused: dict[str, float] = {}
        for mem in (self.bm25_mem, self.dense_mem):
            for r in mem.retrieve(query, depth):
                fused[r.key] = fused.get(r.key, 0.0) + 1.0 / (self.k_rrf + r.rank)
        return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]

    def retrieve(self, query: Query, k: int) -> list[Retrieved]:
        ranked = self._fused(query, k)
        if not ranked:
            return []
        top = ranked[0][1] or 1.0
        return [
            Retrieved(key, score, min(max(score / top, 0.0), 1.0), i)
            for i, (key, score) in enumerate(ranked, 1)
        ]

    def novelty(self, query: Query) -> float:
        return min(self.bm25_mem.novelty(query), self.dense_mem.novelty(query))


_RERANKER: dict = {}


def _reranker(name: str = "BAAI/bge-reranker-base"):
    """Load the BGE cross-encoder reranker once (cached at module level)."""
    if name not in _RERANKER:
        from sentence_transformers import CrossEncoder

        _RERANKER[name] = CrossEncoder(name)
    return _RERANKER[name]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class HybridRerankMemory:
    """Cross-encoder reranking of a hybrid memory's top-n candidates."""

    name = "hybrid_rerank"

    def __init__(
        self,
        hybrid: HybridRRFMemory,
        cross_encoder: str = "BAAI/bge-reranker-base",
        top_n: int = 30,
    ):
        self.hybrid = hybrid
        self.cross_encoder = cross_encoder
        self.top_n = top_n

    def index(self, items: list[IndexItem]) -> None:
        self.hybrid.index(items)
        self._text = {it.key: it.text for it in items}

    def retrieve(self, query: Query, k: int) -> list[Retrieved]:
        candidates = self.hybrid.retrieve(query, self.top_n)
        if not candidates:
            return []
        model = _reranker(self.cross_encoder)
        pairs = [(query.text, self._text.get(c.key, "")) for c in candidates]
        logits = model.predict(pairs)
        scored = sorted(
            zip(candidates, (float(s) for s in logits)),
            key=lambda cs: (-cs[1], cs[0].key),
        )[:k]
        top_conf = _sigmoid(scored[0][1])
        return [
            Retrieved(c.key, logit, top_conf, i)
            for i, (c, logit) in enumerate(scored, 1)
        ]

    def novelty(self, query: Query) -> float:
        return self.hybrid.novelty(query)
