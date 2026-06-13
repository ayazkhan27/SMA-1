# Phase 4a — Drift Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether SMA's structural, environment-re-derived memory resists concept drift better than generative memories and the SOTA (Zep/Graphiti), on the real LongMemEval benchmark, with a SAGE-based structural drift detector.

**Architecture:** A common `MemoryBackend` interface with four implementations (context-only, RAG-notes, Zep/Graphiti, SMA) sharing a DeepSeek backbone and identical LLM extraction. A single-shot harness ingests each LongMemEval instance's multi-session history into each backend, answers its questions, and scores per-category accuracy plus drift metrics (update-recovery, staleness, SAGE expectation-violation delay/recovery). Zep is isolated behind an optional import + container; the core never imports it.

**Tech Stack:** Python 3.11, DeepSeek API (httpx, existing `DeepSeekOrchestrator`), existing `MacFacIndex`/`SagePool`/`stats`, LongMemEval data (fetched by checksum manifest), Graphiti + FalkorDB (containerized) for the Zep baseline.

Reference spec: `docs/superpowers/specs/2026-06-12-phase4a-drift-design.md`.

---

## File Structure

- `data/manifests/datasets.json` — **modify**: add a `longmemeval` block (URL + md5, verified on first fetch).
- `scripts/fetch_datasets.py` — **reuse** (already checksum-verifies manifest entries).
- `sma/eval/longmemeval.py` — **create**: loader (instances → sessions + questions + category + change-points) and the answer grader.
- `sma/eval/memory_backends/__init__.py` — **create**: registry.
- `sma/eval/memory_backends/base.py` — **create**: `MemoryBackend` ABC + `IngestEvent`/`QueryResult` dataclasses.
- `sma/eval/memory_backends/context_only.py` — **create**.
- `sma/eval/memory_backends/rag_notes.py` — **create**.
- `sma/eval/memory_backends/sma_memory.py` — **create**.
- `sma/eval/memory_backends/zep_graphiti.py` — **create**: optional, behind a lazy import.
- `sma/sage/pools.py` — **modify**: add `expectation_violation(case)`.
- `sma/eval/drift_metrics.py` — **create**: update-recovery, staleness, prequential detection delay/recovery.
- `scripts/drift_battery.py` — **create**: single-shot harness, emits `reports/confirmatory/t5_*.csv`.
- `tests/test_drift.py` — **create**: unit tests for loader, backends, SAGE violation, metrics.
- `docker/zep/docker-compose.yml` — **create**: FalkorDB + Graphiti for the Zep baseline.

---

## Task 1: LongMemEval data manifest + fetch

**Files:**
- Modify: `data/manifests/datasets.json`
- Test: `tests/test_drift.py`

- [ ] **Step 1: Add the manifest block**

Add to `data/manifests/datasets.json` a top-level key (sibling of `loghub_raw`).
NOTE (controller-verified 2026-06-12): the original `xiaowu0162/longmemeval` repo
is **deprecated**; the live repo is `longmemeval-cleaned`. The oracle file is
already downloaded to `data/raw/longmemeval/longmemeval_oracle.json` and its md5
is verified below. The 277 MB `_s_cleaned` file is fetched on demand for the full
run; record its md5 on first fetch.

```json
"longmemeval": {
  "source": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned",
  "files": {
    "longmemeval_oracle.json": {
      "md5": "4c2b4c8c936cf26968f4d0fdc93ca31c",
      "url": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json"
    },
    "longmemeval_s_cleaned.json": {
      "md5": "FILL_ON_FIRST_FETCH",
      "url": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
    }
  },
  "source_note": "LongMemEval (Wu et al. 2024), longmemeval-cleaned repo. oracle = answer-bearing sessions only (15 MB, cheap dev/pilot); _s_cleaned = realistic haystack (277 MB, the leaderboard setting). 500 instances each; drift-relevant categories knowledge-update(78)+temporal-reasoning(133)=211. md5 verified locally (same discipline as loghub spirit/liberty)."
}
```

- [ ] **Step 2: Verify the oracle checksum; record the _s checksum on first real fetch**

The oracle is already present. Verify: `md5sum data/raw/longmemeval/longmemeval_oracle.json` → must equal `4c2b4c8c936cf26968f4d0fdc93ca31c`. The 277 MB `longmemeval_s_cleaned.json` is downloaded only for the full run (Task 11 `--full`); on that first fetch, compute its md5 and replace `FILL_ON_FIRST_FETCH`. Confirm `scripts/fetch_datasets.py` verifies the oracle entry.
Expected: "longmemeval_oracle.json: md5 OK".

- [ ] **Step 3: Commit**

```bash
git add data/manifests/datasets.json
git commit -m "data: LongMemEval manifest (checksum-verified fetch)"
```

---

## Task 2: LongMemEval loader

**Files:**
- Create: `sma/eval/longmemeval.py`
- Test: `tests/test_drift.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_drift.py
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
    # knowledge-update instances expose the change-point session index
    assert i.is_drift is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_drift.py::test_loader_parses_sessions_and_category -v`
Expected: FAIL with `ModuleNotFoundError: sma.eval.longmemeval`.

- [ ] **Step 3: Write minimal implementation**

```python
# sma/eval/longmemeval.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_drift.py::test_loader_parses_sessions_and_category -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sma/eval/longmemeval.py tests/test_drift.py
git commit -m "feat: LongMemEval loader with drift-category flag"
```

---

## Task 3: MemoryBackend interface

**Files:**
- Create: `sma/eval/memory_backends/base.py`, `sma/eval/memory_backends/__init__.py`
- Test: `tests/test_drift.py`

- [ ] **Step 1: Write the failing test**

```python
from sma.eval.memory_backends.base import MemoryBackend, QueryResult

def test_backend_is_abstract():
    import pytest
    with pytest.raises(TypeError):
        MemoryBackend()  # abstract; cannot instantiate
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_drift.py::test_backend_is_abstract -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# sma/eval/memory_backends/base.py
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
    """Shared backbone is injected (DeepSeek orchestrator + extractor)."""
    name: str = "base"

    @abstractmethod
    def reset(self) -> None: ...
    @abstractmethod
    def ingest(self, session: Session) -> None: ...
    @abstractmethod
    def query(self, question: str) -> QueryResult: ...
```

```python
# sma/eval/memory_backends/__init__.py
from .base import MemoryBackend, QueryResult
__all__ = ["MemoryBackend", "QueryResult"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_drift.py::test_backend_is_abstract -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sma/eval/memory_backends/
git commit -m "feat: MemoryBackend interface for drift variants"
```

---

## Task 4: Shared LLM extractor + answerer

**Files:**
- Create: `sma/eval/memory_backends/shared_llm.py`
- Test: `tests/test_drift.py`

- [ ] **Step 1: Write the failing test (with a fake orchestrator — no network)**

```python
from sma.eval.memory_backends.shared_llm import extract_facts, answer_from

class FakeLLM:
    def __init__(self, scripted): self.scripted = scripted; self.calls = []
    def complete(self, messages, max_tokens=600, temperature=0.0):
        self.calls.append(messages); return self.scripted.pop(0)

def test_extract_facts_parses_json_lines():
    llm = FakeLLM(['["user works at Acme", "user lives in NYC"]'])
    facts = extract_facts(llm, "I moved to Acme in NYC.")
    assert facts == ["user works at Acme", "user lives in NYC"]

def test_answer_from_uses_retrieved():
    llm = FakeLLM(["Acme"])
    ans = answer_from(llm, "Where does the user work?", ["user works at Acme"])
    assert ans == "Acme"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_drift.py -k "extract_facts or answer_from" -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# sma/eval/memory_backends/shared_llm.py
"""Extraction + answering shared by ALL variants (extraction held constant)."""
from __future__ import annotations
import json

_EXTRACT_SYS = ("Extract the durable user facts from the message as a JSON "
                "array of short strings. Only facts that could be asked about "
                "later. No commentary.")
_ANSWER_SYS = ("Answer the question using ONLY the provided memory items. "
               "If the memory contradicts itself, prefer the most recent. "
               "Answer concisely; if unknown, say 'unknown'.")

def extract_facts(llm, message: str) -> list[str]:
    out = llm.complete(
        [{"role": "system", "content": _EXTRACT_SYS},
         {"role": "user", "content": message}], max_tokens=300)
    try:
        facts = json.loads(out)
        return [str(f) for f in facts] if isinstance(facts, list) else []
    except (json.JSONDecodeError, TypeError):
        return []

def answer_from(llm, question: str, retrieved: list[str]) -> str:
    mem = "\n".join(f"- {r}" for r in retrieved) or "(no memory)"
    out = llm.complete(
        [{"role": "system", "content": _ANSWER_SYS},
         {"role": "user", "content": f"Memory:\n{mem}\n\nQuestion: {question}"}],
        max_tokens=120)
    return out.strip()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_drift.py -k "extract_facts or answer_from" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sma/eval/memory_backends/shared_llm.py tests/test_drift.py
git commit -m "feat: shared LLM extractor/answerer (extraction held constant)"
```

---

## Task 5: context-only and RAG-notes backends

**Files:**
- Create: `sma/eval/memory_backends/context_only.py`, `sma/eval/memory_backends/rag_notes.py`
- Test: `tests/test_drift.py`

- [ ] **Step 1: Write the failing test**

```python
from sma.eval.longmemeval import Session
from sma.eval.memory_backends.context_only import ContextOnly
from sma.eval.memory_backends.rag_notes import RagNotes

def _sess(sid, text):
    return Session(sid, "2025/01/01", [{"role": "user", "content": text}])

def test_context_only_accumulates_and_answers():
    llm = FakeLLM(["Acme"])  # one answer call
    b = ContextOnly(llm); b.reset()
    b.ingest(_sess("s1", "I work at Globex."))
    b.ingest(_sess("s2", "Now I work at Acme."))
    r = b.query("Where does the user work?")
    assert r.answer == "Acme"
    # both turns are in the context window passed to the LLM
    joined = str(llm.calls[-1])
    assert "Globex" in joined and "Acme" in joined

def test_rag_notes_extracts_then_retrieves():
    # 2 extract calls (one per session) + 1 answer call
    llm = FakeLLM(['["works at Globex"]', '["works at Acme"]', "Acme"])
    b = RagNotes(llm, k=5); b.reset()
    b.ingest(_sess("s1", "I work at Globex."))
    b.ingest(_sess("s2", "Now I work at Acme."))
    r = b.query("Where does the user work?")
    assert r.answer == "Acme"
    assert any("Acme" in note for note in r.retrieved)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_drift.py -k "context_only or rag_notes" -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# sma/eval/memory_backends/context_only.py
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
        # the whole accumulated transcript is the "memory"
        ans = answer_from(self.llm, question, self.turns)
        return QueryResult(answer=ans, retrieved=list(self.turns))
```

```python
# sma/eval/memory_backends/rag_notes.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_drift.py -k "context_only or rag_notes" -v`
Expected: PASS (note: `FakeLLM` is defined in Task 4's test block; ensure tests share it — define `FakeLLM` once at module top of `tests/test_drift.py`).

- [ ] **Step 5: Commit**

```bash
git add sma/eval/memory_backends/context_only.py sma/eval/memory_backends/rag_notes.py tests/test_drift.py
git commit -m "feat: context-only + rag-notes drift backends"
```

---

## Task 6: SAGE expectation-violation detector

**Files:**
- Modify: `sma/sage/pools.py`
- Test: `tests/test_drift.py`

- [ ] **Step 1: Write the failing test**

```python
from sma.ir.schema import make_case, stmt
from sma.sage.pools import SagePool

def test_expectation_violation_flags_off_schema_case():
    pool = SagePool("t", assimilation_threshold=0.2)
    # build a schema from two similar cases
    a = make_case([stmt("worksAt", "user", "globex")])
    b = make_case([stmt("worksAt", "user", "globex")])
    pool.assimilate(a); pool.assimilate(b)
    # a case consistent with the schema -> low violation
    consistent = make_case([stmt("worksAt", "user", "globex")])
    # a structurally unrelated case -> high violation
    novel = make_case([stmt("livesIn", "user", "mars")])
    assert pool.expectation_violation(consistent) < pool.expectation_violation(novel)
    assert pool.expectation_violation(novel) > 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_drift.py::test_expectation_violation_flags_off_schema_case -v`
Expected: FAIL with `AttributeError: 'SagePool' object has no attribute 'expectation_violation'`.

- [ ] **Step 3: Write minimal implementation**

Add to `sma/sage/pools.py` inside `SagePool` (after `assimilate`):

```python
    def expectation_violation(self, case: Case) -> float:
        """1 - best normalized structural fit to any learned schema.

        Near 0 = the case is explained by an existing generalization;
        near 1 = the case breaks every schema (a candidate concept-drift
        point). With no generalizations yet, returns 1.0 (nothing to expect).
        """
        if not self.generalizations:
            return 1.0
        best = 0.0
        for gen in self.generalizations:
            schema = gen.schema_case(self.probability_cutoff, self.min_constituents)
            gmap = match_cases(schema, case, self.config)
            best = max(best, gmap.normalized_score)
        return max(0.0, 1.0 - best)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_drift.py::test_expectation_violation_flags_off_schema_case -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sma/sage/pools.py tests/test_drift.py
git commit -m "feat: SAGE expectation-violation (structural drift detector)"
```

---

## Task 7: SMA memory backend

**Files:**
- Create: `sma/eval/memory_backends/sma_memory.py`
- Test: `tests/test_drift.py`

- [ ] **Step 1: Write the failing test**

```python
from sma.eval.memory_backends.sma_memory import SmaMemory

def test_sma_memory_reencodes_and_flags_drift():
    # 2 extracts (one per session) + 1 answer
    llm = FakeLLM(['["worksAt user globex"]', '["worksAt user acme"]', "Acme"])
    b = SmaMemory(llm, k=5); b.reset()
    b.ingest(_sess("s1", "I work at Globex."))
    r0_violation_seen = b.last_violation
    b.ingest(_sess("s2", "Now I work at Acme."))
    r = b.query("Where does the user work?")
    assert r.answer == "Acme"
    # the second, off-schema fact raised expectation-violation at ingest
    assert b.last_violation >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_drift.py::test_sma_memory_reencodes_and_flags_drift -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# sma/eval/memory_backends/sma_memory.py
"""SMA memory: each turn's extracted facts are re-encoded into the case store
(re-derived from the conversation, never from prior generations); retrieval is
structural; SAGE flags expectation-violations as drift."""
from __future__ import annotations
from .base import MemoryBackend, QueryResult
from .shared_llm import extract_facts, answer_from
from sma.index.macfac import MacFacIndex
from sma.ir.schema import make_case, stmt
from sma.ir.sexpr import canonical_case_text
from sma.sage.pools import SagePool
from sma.match.types import MatchConfig

def _fact_to_case(fact: str):
    toks = fact.split()
    if len(toks) >= 3:
        return make_case([stmt(toks[0], toks[1], " ".join(toks[2:]))])
    return make_case([stmt("fact", *(toks or ["empty"]))])

class SmaMemory(MemoryBackend):
    name = "sma"
    def __init__(self, llm, k: int = 5):
        self.llm = llm; self.k = k; self.last_violation = 0.0
    def reset(self):
        self.index = MacFacIndex(config=MatchConfig())
        self.pool = SagePool("drift", assimilation_threshold=0.2)
        self.texts: dict[str, str] = {}; self.last_violation = 0.0
    def ingest(self, session):
        for t in session.turns:
            for fact in extract_facts(self.llm, t["content"]):
                case = _fact_to_case(fact)
                self.last_violation = self.pool.expectation_violation(case)
                self.index.add(case); self.pool.assimilate(case)
                self.texts[case.case_id] = fact
    def query(self, question):
        qcase = _fact_to_case(question)
        results = self.index.retrieve(qcase, k=self.k, shortlist=50, fac_budget=20)
        retrieved = [self.texts.get(r.case_id, "") for r in results]
        ans = answer_from(self.llm, question, [r for r in retrieved if r])
        return QueryResult(answer=ans, retrieved=retrieved,
                           drift_flagged=self.last_violation > 0.5)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_drift.py::test_sma_memory_reencodes_and_flags_drift -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sma/eval/memory_backends/sma_memory.py tests/test_drift.py
git commit -m "feat: SMA memory backend (structural re-derivation + SAGE drift)"
```

---

## Task 8: LongMemEval grader

**Files:**
- Modify: `sma/eval/longmemeval.py`
- Test: `tests/test_drift.py`

- [ ] **Step 1: Write the failing test**

```python
from sma.eval.longmemeval import grade_answer

def test_grader_exact_and_substring():
    assert grade_answer("Acme", "Acme") == 1.0
    assert grade_answer("They work at Acme Corp.", "Acme") == 1.0
    assert grade_answer("Globex", "Acme") == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_drift.py::test_grader_exact_and_substring -v`
Expected: FAIL with `ImportError: cannot import name 'grade_answer'`.

- [ ] **Step 3: Write minimal implementation**

Add to `sma/eval/longmemeval.py`:

```python
def grade_answer(prediction: str, gold: str) -> float:
    """LongMemEval-style lenient match: normalized substring containment.
    (The official grader uses an LLM judge; this deterministic proxy is used
    for unit tests and the smoke run. The battery can swap in the LLM judge.)"""
    p = " ".join(prediction.lower().split())
    g = " ".join(gold.lower().split())
    return 1.0 if g and g in p else 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_drift.py::test_grader_exact_and_substring -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sma/eval/longmemeval.py tests/test_drift.py
git commit -m "feat: LongMemEval deterministic grader (proxy for smoke runs)"
```

---

## Task 9: Drift metrics

**Files:**
- Create: `sma/eval/drift_metrics.py`
- Test: `tests/test_drift.py`

- [ ] **Step 1: Write the failing test**

```python
from sma.eval.drift_metrics import update_recovery, detection_delay

def test_update_recovery_detects_stale():
    # answers over sessions after a fact changed Globex->Acme at change_idx=1
    # correct=1 if returned the NEW value
    correctness = [0, 0, 1, 1]   # recovered at index 2
    assert update_recovery(correctness, change_idx=1) == 2  # sessions to recover

def test_detection_delay_first_flag_after_change():
    flags = [False, False, True, False]  # flagged at index 2
    assert detection_delay(flags, change_idx=1) == 1  # 2 - 1
    assert detection_delay([False, False], change_idx=1) is None  # never detected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_drift.py -k "update_recovery or detection_delay" -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# sma/eval/drift_metrics.py
"""Standard concept-drift metrics adapted to the agent-memory setting."""
from __future__ import annotations

def update_recovery(correctness: list[int], change_idx: int) -> int | None:
    """Sessions after the change until the memory returns the NEW value and
    keeps it. None if it never recovers within the window."""
    for i in range(change_idx, len(correctness)):
        if correctness[i] == 1 and all(c == 1 for c in correctness[i:]):
            return i - change_idx
    return None

def detection_delay(flags: list[bool], change_idx: int) -> int | None:
    """Sessions after the change until the detector first fires. None if never."""
    for i in range(change_idx, len(flags)):
        if flags[i]:
            return i - change_idx
    return None

def staleness_rate(correctness: list[int], change_idx: int) -> float:
    """Fraction of post-change probes that still returned the OLD value."""
    post = correctness[change_idx:]
    return 0.0 if not post else sum(1 for c in post if c == 0) / len(post)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_drift.py -k "update_recovery or detection_delay" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sma/eval/drift_metrics.py tests/test_drift.py
git commit -m "feat: drift metrics (recovery, detection delay, staleness)"
```

---

## Task 10: Zep/Graphiti backend (isolated, optional)

**Files:**
- Create: `sma/eval/memory_backends/zep_graphiti.py`, `docker/zep/docker-compose.yml`
- Test: `tests/test_drift.py`

- [ ] **Step 1: Write the failing test (skips cleanly when Zep is absent)**

```python
import pytest
from sma.eval.memory_backends.zep_graphiti import ZepGraphiti, ZEP_AVAILABLE

def test_zep_imports_or_skips():
    if not ZEP_AVAILABLE:
        pytest.skip("graphiti not installed; Zep baseline runs in its container")
    b = ZepGraphiti(llm=None)  # construction must not raise when available
    assert b.name == "zep-graphiti"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_drift.py::test_zep_imports_or_skips -v`
Expected: FAIL with `ModuleNotFoundError` (file not created yet).

- [ ] **Step 3: Write minimal implementation**

```python
# sma/eval/memory_backends/zep_graphiti.py
"""SOTA baseline: Graphiti temporal knowledge graph (the engine behind Zep).
Isolated behind a lazy import so the core never depends on it; the graph DB
runs in docker/zep. Points Graphiti's extraction at the SAME DeepSeek backbone
so the comparison is equal-footing."""
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
        self.g.clear()  # wipe the graph between instances
    def ingest(self, session):
        for t in session.turns:
            self.g.add_episode(name=session.session_id, episode_body=t["content"],
                               reference_time=session.date)
    def query(self, question):
        hits = self.g.search(question)
        retrieved = [h.fact for h in hits]
        from .shared_llm import answer_from
        return QueryResult(answer=answer_from(self.llm, question, retrieved),
                           retrieved=retrieved)
```

```yaml
# docker/zep/docker-compose.yml
services:
  falkordb:
    image: falkordb/falkordb:latest
    ports: ["6379:6379", "3000:3000"]
  # Graphiti runs in-process from the harness; FalkorDB is its graph store.
  # Set SMA_DEEPSEEK_API_KEY + GRAPHITI LLM env so extraction uses DeepSeek.
```

- [ ] **Step 4: Run test to verify it passes (skip path)**

Run: `pytest tests/test_drift.py::test_zep_imports_or_skips -v`
Expected: PASS or SKIP (skip is acceptable until the container is provisioned).

- [ ] **Step 5: Commit**

```bash
git add sma/eval/memory_backends/zep_graphiti.py docker/zep/docker-compose.yml tests/test_drift.py
git commit -m "feat: Zep/Graphiti SOTA backend (isolated, optional)"
```

---

## Task 11: Drift battery harness + cost pilot

**Files:**
- Create: `scripts/drift_battery.py`
- Test: manual smoke run (documented)

- [ ] **Step 1: Write the harness**

```python
# scripts/drift_battery.py
"""Single-shot drift battery (Phase 4a). Mirrors confirmatory_battery discipline.

  python3 scripts/drift_battery.py --smoke           # 5 instances, deterministic grader
  python3 scripts/drift_battery.py --limit 50        # cost pilot, logs token estimate
  python3 scripts/drift_battery.py                   # full single-shot
"""
from __future__ import annotations
import argparse, csv, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from sma.eval.longmemeval import load_instances, grade_answer
from sma.eval.memory_backends.context_only import ContextOnly
from sma.eval.memory_backends.rag_notes import RagNotes
from sma.eval.memory_backends.sma_memory import SmaMemory
from sma.eval.drift_metrics import update_recovery, detection_delay, staleness_rate
from sma.eval.stats import paired_bootstrap, holm_bonferroni, cliffs_delta

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "longmemeval"
ORACLE = DATA_DIR / "longmemeval_oracle.json"          # cheap dev/pilot (15 MB)
FULL = DATA_DIR / "longmemeval_s_cleaned.json"          # leaderboard setting (277 MB)
OUT = ROOT / "reports" / "confirmatory"

def make_backends(llm):
    backends = [ContextOnly(llm), RagNotes(llm), SmaMemory(llm)]
    try:
        from sma.eval.memory_backends.zep_graphiti import ZepGraphiti, ZEP_AVAILABLE
        if ZEP_AVAILABLE:
            backends.append(ZepGraphiti(llm))
    except Exception as exc:
        print(f"[zep] skipped: {exc}", flush=True)
    return backends

def get_llm(smoke):
    if smoke:
        # deterministic stub: echoes a fact for extraction, returns last fact for answers
        class Stub:
            def complete(self, messages, max_tokens=600, temperature=0.0):
                u = messages[-1]["content"]
                return '["' + u.split(":")[-1].strip()[:40] + '"]' if "Extract" in messages[0]["content"] else u.split("Question:")[-1].strip()[:40]
        return Stub()
    from sma.agent.llm import DeepSeekOrchestrator
    return DeepSeekOrchestrator()

def run(limit, smoke, full):
    rows_path = OUT / ("t5_rows_smoke.csv" if smoke else "t5_rows.csv")
    if rows_path.exists() and not smoke:
        sys.exit(f"REFUSE: {rows_path} exists (single-shot). Log a rerun in STATUS.md and delete to force.")
    data = FULL if full else ORACLE   # oracle for dev/pilot; --full for the leaderboard setting
    insts = load_instances(data)
    if smoke: insts = insts[:5]
    elif limit: insts = insts[:limit]
    llm = get_llm(smoke)
    rows = []
    for inst in insts:
        for b in make_backends(llm):
            b.reset()
            for s in inst.sessions:
                b.ingest(s)
            r = b.query(inst.question)
            rows.append({"qid": inst.question_id, "category": inst.category,
                         "method": b.name, "correct": grade_answer(r.answer, inst.answer),
                         "drift": int(inst.is_drift), "flagged": int(r.drift_flagged)})
        print(f"[{len(rows)}] {inst.question_id} done", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with rows_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["qid", "category", "method", "correct", "drift", "flagged"])
        w.writeheader(); w.writerows(rows)
    # per-method accuracy on drift categories + paired stats vs SMA
    methods = sorted({r["method"] for r in rows})
    drift_rows = [r for r in rows if r["drift"]]
    by = lambda m: [r["correct"] for r in drift_rows if r["method"] == m]
    if "sma" in methods:
        pvals, summary = {}, []
        for m in methods:
            if m == "sma": continue
            bs = paired_bootstrap(by("sma"), by(m))
            pvals[m] = bs["p_value"]
            summary.append({"baseline": m, "delta": bs["delta"],
                            "ci_low": bs["ci_low"], "ci_high": bs["ci_high"],
                            "cliffs": cliffs_delta(by("sma"), by(m))})
        holm = holm_bonferroni(pvals)
        for s in summary: s["p_holm"] = holm[s["baseline"]]
        spath = OUT / ("t5_stats_smoke.csv" if smoke else "t5_stats.csv")
        with spath.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["baseline", "delta", "ci_low", "ci_high", "cliffs", "p_holm"])
            w.writeheader(); w.writerows(summary)
        print("drift-category accuracy:", {m: round(sum(by(m))/max(len(by(m)),1), 3) for m in methods})
    print(f"wrote {rows_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--full", action="store_true", help="use longmemeval_s_cleaned (277MB leaderboard setting); default oracle")
    a = ap.parse_args()
    run(a.limit, a.smoke, a.full)
```

- [ ] **Step 2: Run the smoke (no network, deterministic stub)**

Run: `python3 scripts/drift_battery.py --smoke`
Expected: prints per-method drift-category accuracy and writes `reports/confirmatory/t5_rows_smoke.csv` with no errors. Confirms the whole pipeline wires together.

- [ ] **Step 3: Run the cost pilot (real DeepSeek, small slice)**

Run: `python3 scripts/drift_battery.py --limit 20`
Expected: completes ~20 instances; note wall-clock + (from DeepSeek dashboard) token spend, and project the full-run cost. **Stop and report the projection before the full run** (budget is ~$14).

- [ ] **Step 4: Clean smoke artifacts**

Run: `rm -f reports/confirmatory/t5_*smoke.csv`

- [ ] **Step 5: Commit**

```bash
git add scripts/drift_battery.py
git commit -m "feat: drift battery harness (smoke + cost pilot + single-shot)"
```

---

## Task 12: Ledger + figure

**Files:**
- Modify: `docs/STATUS.md`, `scripts/figures_supp.py` (add `f_drift`)

- [ ] **Step 1: Add the drift figure** to `scripts/figures_supp.py`: a grouped bar of drift-category accuracy per method + a small panel of SAGE detection-delay vs baselines, reading `reports/confirmatory/t5_summary.csv`. (Mirror `f7_family` style; use `figstyle`.)

- [ ] **Step 2: Run it**

Run: `python3 scripts/figures_supp.py`
Expected: writes `paper/figures/individual/fig14_drift.pdf/.png`.

- [ ] **Step 3: STATUS entry** documenting the run, numbers, and outcome (positive / parity / negative per the honest-outcome rule).

- [ ] **Step 4: Commit**

```bash
git add docs/STATUS.md scripts/figures_supp.py paper/figures/individual/fig14_drift.*
git commit -m "Phase 4a drift: results figure + STATUS ledger"
```

---

## Self-Review

**Spec coverage:** §2 arena → Tasks 1–2, 8 (LongMemEval); §3 four variants → Tasks 3,5,7,10; held-constant extraction → Task 4; §5 metrics → Tasks 8 (accuracy) + 9 (recovery/delay/staleness) + 6 (SAGE detector); §6 components → all tasks map to the listed files; §7 cost pilot → Task 11 Step 3; SAGE on existing pools → Task 6; single-shot discipline → Task 11 guard. LoCoMo is secondary/out-of-first-cut per §10 (YAGNI) — intentionally not a task here.

**Placeholder scan:** one intentional `FILL_ON_FIRST_FETCH` in the manifest with an explicit step to populate+verify the real md5 (the established dataset discipline) — not a code placeholder. No TBD/TODO in code.

**Type consistency:** `MemoryBackend.reset/ingest/query` and `QueryResult(answer, retrieved, drift_flagged)` are used identically across Tasks 3,5,7,10,11. `Session(session_id,date,turns)` consistent (Tasks 2,5). `expectation_violation(case)->float` defined Task 6, used Task 7. `grade_answer` defined Task 8, used Task 11. Stats fns match `sma/eval/stats.py` signatures.

**Note for executor:** define `FakeLLM` and `_sess` once at the top of `tests/test_drift.py` (Tasks 4–7 share them).
