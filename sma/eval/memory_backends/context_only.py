from __future__ import annotations
from .base import MemoryBackend, QueryResult
from .shared_llm import answer_from

class ContextOnly(MemoryBackend):
    name = "context-only"
    def __init__(self, llm): self.llm = llm; self.turns: list[str] = []
    def reset(self): self.turns = []
    def ingest(self, session):
        for t in session.turns:
            self.turns.append(f"[{session.date}] {t['content']}")
    def query(self, question):
        ans = answer_from(self.llm, question, self.turns)
        return QueryResult(answer=ans, retrieved=list(self.turns))
