from __future__ import annotations

import json
from sma.eval.longmemeval import load_instances, LMEInstance

def _fixture(tmp_path):
    data = [{
        "question_id": "q1", "question_type": "knowledge-update",
        "question": "Where does the user work now?", "answer": "Acme",
        "question_date": "2025/03/01",
        "haystack_session_ids": ["s1", "s2"],
        "haystack_dates": ["2025/01/01", "2025/02/01"],
        "haystack_sessions": [
            [{"role": "user", "content": "I work at Globex."}],
            [{"role": "user", "content": "I changed jobs, now at Acme."}],
        ],
        "answer_session_ids": ["s2"],
    }]
    p = tmp_path / "lme.json"; p.write_text(json.dumps(data)); return p

def test_loader_parses_sessions_and_category(tmp_path):
    inst = load_instances(_fixture(tmp_path))
    assert len(inst) == 1
    i = inst[0]
    assert isinstance(i, LMEInstance)
    assert i.category == "knowledge-update"
    assert len(i.sessions) == 2
    assert i.sessions[1].turns[0]["content"].startswith("I changed jobs")
    assert i.answer == "Acme"
    assert i.is_drift is True


import pathlib
import pytest

ORACLE = pathlib.Path("data/raw/longmemeval/longmemeval_oracle.json")

@pytest.mark.skipif(not ORACLE.exists(), reason="LongMemEval oracle not fetched")
def test_loader_on_real_oracle():
    inst = load_instances(ORACLE)
    assert len(inst) == 500
    assert any(i.is_drift for i in inst)
    cats = {i.category for i in inst}
    assert "knowledge-update" in cats and "temporal-reasoning" in cats


from sma.eval.memory_backends.base import MemoryBackend, QueryResult


# ---------------------------------------------------------------------------
# Shared helper — used by Task 4 and all subsequent drift-experiment tests
# ---------------------------------------------------------------------------
class FakeLLM:
    """Scripted stand-in for DeepSeek: pops the next canned response per call."""
    def __init__(self, scripted): self.scripted = list(scripted); self.calls = []
    def complete(self, messages, max_tokens=600, temperature=0.0):
        self.calls.append(messages)
        return self.scripted.pop(0)


def test_backend_is_abstract():
    import pytest
    with pytest.raises(TypeError):
        MemoryBackend()  # abstract; cannot instantiate

def test_query_result_defaults():
    r = QueryResult(answer="x")
    assert r.retrieved == [] and r.drift_flagged is False


def test_extract_facts_parses_json_array():
    from sma.eval.memory_backends.shared_llm import extract_facts
    llm = FakeLLM(['["user works at Acme", "user lives in NYC"]'])
    facts = extract_facts(llm, "I moved to Acme in NYC.")
    assert facts == ["user works at Acme", "user lives in NYC"]

def test_extract_facts_handles_bad_json():
    from sma.eval.memory_backends.shared_llm import extract_facts
    llm = FakeLLM(["not json at all"])
    assert extract_facts(llm, "whatever") == []

def test_answer_from_uses_retrieved():
    from sma.eval.memory_backends.shared_llm import answer_from
    llm = FakeLLM(["Acme"])
    ans = answer_from(llm, "Where does the user work?", ["user works at Acme"])
    assert ans == "Acme"
    # the retrieved memory must appear in the prompt sent to the LLM
    assert "user works at Acme" in str(llm.calls[-1])


from sma.eval.longmemeval import Session

def _sess(sid, text):
    return Session(sid, "2025/01/01", [{"role": "user", "content": text}])

def test_context_only_accumulates_and_answers():
    from sma.eval.memory_backends.context_only import ContextOnly
    llm = FakeLLM(["Acme"])  # one answer call
    b = ContextOnly(llm); b.reset()
    b.ingest(_sess("s1", "I work at Globex."))
    b.ingest(_sess("s2", "Now I work at Acme."))
    r = b.query("Where does the user work?")
    assert r.answer == "Acme"
    joined = str(llm.calls[-1])
    assert "Globex" in joined and "Acme" in joined  # full transcript is the memory

def test_rag_notes_extracts_then_retrieves():
    from sma.eval.memory_backends.rag_notes import RagNotes
    # 2 extract calls (one per session turn) + 1 answer call
    llm = FakeLLM(['["works at Globex"]', '["works at Acme"]', "Acme"])
    b = RagNotes(llm, k=5); b.reset()
    b.ingest(_sess("s1", "I work at Globex."))
    b.ingest(_sess("s2", "Now I work at Acme."))
    r = b.query("Where does the user work?")
    assert r.answer == "Acme"
    assert any("Acme" in note for note in r.retrieved)


def test_expectation_violation_flags_off_schema_case():
    from sma.ir.schema import make_case, stmt
    from sma.sage.pools import SagePool
    pool = SagePool("t", assimilation_threshold=0.2)
    # build a schema from two similar cases
    a = make_case([stmt("worksAt", "user", "globex")])
    b = make_case([stmt("worksAt", "user", "globex")])
    pool.assimilate(a); pool.assimilate(b)
    consistent = make_case([stmt("worksAt", "user", "globex")])
    novel = make_case([stmt("livesIn", "user", "mars")])
    assert pool.expectation_violation(consistent) < pool.expectation_violation(novel)
    assert pool.expectation_violation(novel) > 0.5

def test_expectation_violation_empty_pool_is_max():
    from sma.ir.schema import make_case, stmt
    from sma.sage.pools import SagePool
    pool = SagePool("t2", assimilation_threshold=0.2)
    assert pool.expectation_violation(make_case([stmt("x", "a", "b")])) == 1.0


def test_sma_memory_reencodes_and_flags_drift():
    from sma.eval.memory_backends.sma_memory import SmaMemory
    # 2 extracts (one per session turn) + 1 answer
    llm = FakeLLM(['["worksAt user globex"]', '["worksAt user acme"]', "Acme"])
    b = SmaMemory(llm, k=5); b.reset()
    b.ingest(_sess("s1", "I work at Globex."))
    b.ingest(_sess("s2", "Now I work at Acme."))
    r = b.query("worksAt user ?")
    assert r.answer == "Acme"
    assert b.last_violation >= 0.0   # expectation-violation tracked at ingest
    assert isinstance(r.drift_flagged, bool)


def test_grader_exact_and_substring():
    from sma.eval.longmemeval import grade_answer
    assert grade_answer("Acme", "Acme") == 1.0
    assert grade_answer("They work at Acme Corp.", "Acme") == 1.0
    assert grade_answer("Globex", "Acme") == 0.0
    assert grade_answer("", "Acme") == 0.0


def test_update_recovery_detects_recovery_point():
    from sma.eval.drift_metrics import update_recovery
    # delta convention: sessions AFTER change_idx until stable recovery (0 = at change)
    assert update_recovery([0, 0, 1, 1], change_idx=1) == 1   # recovers at index 2 -> 1 after change
    assert update_recovery([0, 0, 0, 0], change_idx=1) is None
    assert update_recovery([0, 1, 0, 1], change_idx=1) == 2   # stable from index 3 -> 2 after change

def test_detection_delay():
    from sma.eval.drift_metrics import detection_delay
    assert detection_delay([False, False, True, False], change_idx=1) == 1  # 2 - 1
    assert detection_delay([False, False], change_idx=1) is None  # never detected

def test_staleness_rate():
    from sma.eval.drift_metrics import staleness_rate
    assert staleness_rate([1, 0, 0, 1], change_idx=1) == 2/3  # post-change: [0,0,1] -> 2 stale of 3
    assert staleness_rate([1], change_idx=1) == 0.0  # no post-change probes
