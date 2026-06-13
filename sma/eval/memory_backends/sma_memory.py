"""SMA memory: each turn's extracted facts are re-encoded into the case store
(re-derived from the conversation, never from prior generations); retrieval is
structural; SAGE flags expectation-violations as drift."""
from __future__ import annotations
from .base import MemoryBackend, QueryResult
from .shared_llm import extract_facts, answer_from
from sma.index.macfac import MacFacIndex
from sma.ir.schema import make_case, stmt
from sma.sage.pools import SagePool
from sma.match.types import MatchConfig


def _fact_to_case(fact: str):
    toks = fact.split()
    if len(toks) >= 3:
        return make_case([stmt(toks[0], toks[1], " ".join(toks[2:]))])
    return make_case([stmt("fact", *(toks or ["empty"]))])


class SmaMemory(MemoryBackend):
    """SMA memory backend: structural re-derivation per turn + SAGE drift detection."""

    name = "sma"

    def __init__(self, llm, k: int = 5):
        self.llm = llm
        self.k = k
        self.last_violation = 0.0

    def reset(self):
        self.index = MacFacIndex(config=MatchConfig())
        self.pool = SagePool("drift", assimilation_threshold=0.2)
        self.texts: dict[str, str] = {}
        self.last_violation = 0.0

    def ingest(self, session) -> None:
        for t in session.turns:
            for fact in extract_facts(self.llm, t["content"]):
                case = _fact_to_case(fact)
                self.last_violation = self.pool.expectation_violation(case)
                self.index.add(case)
                self.pool.assimilate(case)
                self.texts[case.case_id] = fact

    def query(self, question: str) -> QueryResult:
        qcase = _fact_to_case(question)
        results = self.index.retrieve(qcase, k=self.k, shortlist=50, fac_budget=20)
        retrieved = [self.texts.get(r.case_id, "") for r in results]
        ans = answer_from(self.llm, question, [r for r in retrieved if r])
        return QueryResult(
            answer=ans,
            retrieved=retrieved,
            drift_flagged=self.last_violation > 0.5,
        )
