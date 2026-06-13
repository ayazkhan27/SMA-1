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

def test_backend_is_abstract():
    import pytest
    with pytest.raises(TypeError):
        MemoryBackend()  # abstract; cannot instantiate

def test_query_result_defaults():
    r = QueryResult(answer="x")
    assert r.retrieved == [] and r.drift_flagged is False
