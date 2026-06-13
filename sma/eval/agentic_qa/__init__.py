"""Phase 5 LLM-QA harness: the one-shot agent + trustworthy-specialist metrics."""

from __future__ import annotations

from sma.eval.agentic_qa.agent import MockLLM, QAAgent
from sma.eval.agentic_qa.metrics import (
    abstention,
    accuracy,
    citation_faithfulness,
    novelty_recall,
)
from sma.eval.agentic_qa.pools import QAItem, build_pools

__all__ = [
    "QAAgent",
    "MockLLM",
    "QAItem",
    "build_pools",
    "accuracy",
    "citation_faithfulness",
    "abstention",
    "novelty_recall",
]
