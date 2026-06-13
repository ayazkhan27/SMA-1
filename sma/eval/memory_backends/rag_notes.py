from __future__ import annotations
from .base import MemoryBackend, QueryResult
from .shared_llm import extract_facts, answer_from

class RagNotes(MemoryBackend):
    """LLM-written notes, retrieved by token overlap (a faithful simple RAG)."""
    name = "rag-notes"
    def __init__(self, llm, k: int = 5): self.llm = llm; self.k = k; self.notes: list[str] = []
    def reset(self): self.notes = []
    def ingest(self, session):
        for t in session.turns:
            self.notes.extend(extract_facts(self.llm, t["content"]))
    def query(self, question):
        q = set(question.lower().split())
        ranked = sorted(self.notes, key=lambda n: -len(q & set(n.lower().split())))
        top = ranked[: self.k]
        return QueryResult(answer=answer_from(self.llm, question, top), retrieved=top)
