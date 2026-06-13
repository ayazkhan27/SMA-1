"""SOTA baseline: Graphiti temporal knowledge graph (the engine behind Zep).
Isolated behind a lazy import so the core never depends on it; the graph DB
runs in docker/zep. Graphiti's extraction is pointed at the SAME DeepSeek
backbone (via env) so the comparison is equal-footing."""
from __future__ import annotations
from .base import MemoryBackend, QueryResult

try:
    import graphiti_core  # noqa: F401
    ZEP_AVAILABLE = True
except Exception:
    ZEP_AVAILABLE = False

class ZepGraphiti(MemoryBackend):
    name = "zep-graphiti"
    def __init__(self, llm, uri: str = "bolt://localhost:7687"):
        if not ZEP_AVAILABLE:
            raise RuntimeError("graphiti_core not installed; see docker/zep/README")
        from graphiti_core import Graphiti
        self.g = Graphiti(uri)   # configured to use DeepSeek via env in the container
        self.llm = llm
    def reset(self):
        self.g.clear()
    def ingest(self, session):
        for t in session.turns:
            self.g.add_episode(name=session.session_id, episode_body=t["content"],
                               reference_time=session.date)
    def query(self, question):
        from .shared_llm import answer_from
        hits = self.g.search(question)
        retrieved = [h.fact for h in hits]
        return QueryResult(answer=answer_from(self.llm, question, retrieved),
                           retrieved=retrieved)
