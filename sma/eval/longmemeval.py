"""LongMemEval loader + answer grader (real agent-memory drift benchmark)."""
from __future__ import annotations
import json, pathlib
from dataclasses import dataclass

DRIFT_CATEGORIES = {"knowledge-update", "temporal-reasoning"}

@dataclass(frozen=True)
class Session:
    session_id: str
    date: str
    turns: list[dict]

@dataclass(frozen=True)
class LMEInstance:
    question_id: str
    category: str
    question: str
    answer: str
    question_date: str
    sessions: tuple[Session, ...]
    answer_session_ids: tuple[str, ...]

    @property
    def is_drift(self) -> bool:
        return self.category in DRIFT_CATEGORIES

def load_instances(path: str | pathlib.Path) -> list[LMEInstance]:
    raw = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    out: list[LMEInstance] = []
    for r in raw:
        sessions = tuple(
            Session(sid, date, turns)
            for sid, date, turns in zip(
                r["haystack_session_ids"], r["haystack_dates"], r["haystack_sessions"])
        )
        out.append(LMEInstance(
            question_id=r["question_id"], category=r["question_type"],
            question=r["question"], answer=str(r["answer"]),
            question_date=r.get("question_date", ""), sessions=sessions,
            answer_session_ids=tuple(r.get("answer_session_ids", []))))
    return out


def grade_answer(prediction: str, gold: str) -> float:
    """LongMemEval-style lenient match: normalized substring containment.
    (The official grader uses an LLM judge; this deterministic proxy is used
    for unit tests and the smoke run. The battery can swap in the LLM judge.)"""
    p = " ".join(prediction.lower().split())
    g = " ".join(gold.lower().split())
    return 1.0 if g and g in p else 0.0
