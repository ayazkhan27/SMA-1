"""Common interface for the four drift-experiment memory variants."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from sma.eval.longmemeval import Session

@dataclass
class QueryResult:
    answer: str
    retrieved: list[str] = field(default_factory=list)
    drift_flagged: bool = False   # backend believes the queried fact changed

class MemoryBackend(ABC):
    """Shared backbone (DeepSeek orchestrator + extractor) is injected by the harness."""
    name: str = "base"

    @abstractmethod
    def reset(self) -> None: ...
    @abstractmethod
    def ingest(self, session: Session) -> None: ...
    @abstractmethod
    def query(self, question: str) -> QueryResult: ...
