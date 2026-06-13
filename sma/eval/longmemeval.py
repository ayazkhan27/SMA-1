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
