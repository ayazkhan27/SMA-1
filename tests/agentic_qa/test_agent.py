"""Tests for the Phase 5 LLM-QA agent + driver (sma.eval.agentic_qa.agent,
scripts/agentic_qa.py).

Everything runs against a tiny toy memory + toy ``QAItem``s with a deterministic
:class:`MockLLM`, so NO real DeepSeek call is ever made (the orchestrator is not
even imported on the ``--mock`` path). Covers:

* a well-formed result dict for each registered condition (grounded SMA-like /
  grounded dense-like / closed-book), with every field the metrics read;
* abstain parsing (``{"choice": 0}`` and closed-book ``ABSTAIN``);
* defensive JSON parsing (code fences, noisy prose, garbage -> abstain);
* the SMA novelty flag honouring ``novelty_threshold``;
* the driver running end-to-end under ``--mock`` on a tiny n, writing both CSVs,
  with the real-LLM constructor patched to explode if ever touched.
"""

from __future__ import annotations

import csv
import importlib.util
import pathlib

import pytest

from sma.eval.agentic import Query, Retrieved
from sma.eval.agentic_qa.agent import MockLLM, QAAgent, _parse_json_object
from sma.eval.agentic_qa.pools import QAItem

RESULT_FIELDS = {
    "gold_id",
    "gold_name",
    "answerable",
    "novel",
    "abstained",
    "pred_id",
    "answer",
    "novelty_flag",
    "confidence",
}

KEY_TO_NAME = {"OMIM:1": "Disease One", "OMIM:2": "Disease Two", "OMIM:3": "Disease Three"}
KEY_TO_TERMS = {
    "OMIM:1": frozenset({"HP:0001001", "HP:0001002"}),
    "OMIM:2": frozenset({"HP:0002001"}),
    "OMIM:3": frozenset({"HP:0003001"}),
}


class ToyMemory:
    """A minimal in-test ``Memory``: canned ranked hits + a fixed novelty score.

    Honours the frozen three-method contract (``index`` / ``retrieve`` /
    ``novelty``) so the agent treats it exactly like ``SmaMemory`` / ``DenseMemory``.
    ``mounted`` is ``None`` so the agent falls back to ``key_to_terms`` ids when
    rendering features (the real SMA memory exposes ``.mounted`` for term names).
    """

    name = "toy"
    mounted = None

    def __init__(self, retrieved: list[Retrieved], novelty: float = 0.0):
        self._retrieved = retrieved
        self._novelty = novelty
        self.indexed: list | None = None

    def index(self, items: list) -> None:
        self.indexed = items

    def retrieve(self, query: Query, k: int) -> list[Retrieved]:
        return self._retrieved[:k]

    def novelty(self, query: Query) -> float:
        return self._novelty


def _item(*, answerable: bool = True, novel: bool = False, gold_id: str = "OMIM:1") -> QAItem:
    return QAItem(
        case_text="Patient presents with: phenotype one, phenotype two",
        case_terms=frozenset({"HP:0001001", "HP:0001002"}),
        gold_id=gold_id,
        gold_name=KEY_TO_NAME[gold_id],
        answerable=answerable,
        novel=novel,
    )


def _agent(llm, memory, **kw) -> QAAgent:
    params = dict(key_to_name=KEY_TO_NAME, key_to_terms=KEY_TO_TERMS, k=5)
    params.update(kw)
    return QAAgent(llm, memory, **params)


def _two_hits() -> list[Retrieved]:
    return [
        Retrieved("OMIM:1", 0.91, 0.80, 1),
        Retrieved("OMIM:2", 0.40, 0.80, 2),
    ]


# --- defensive JSON parsing ------------------------------------------------
def test_parse_json_plain_fenced_and_noisy():
    assert _parse_json_object('{"choice": 2}') == {"choice": 2}
    assert _parse_json_object('```json\n{"choice": 0}\n```') == {"choice": 0}
    assert _parse_json_object('```\n{"diagnosis": "X"}\n```') == {"diagnosis": "X"}
    # leading/trailing prose around a single object still parses.
    assert _parse_json_object('Sure: {"diagnosis": "X"} done') == {"diagnosis": "X"}
    # no object at all -> None.
    assert _parse_json_object("no json here") is None


# --- grounded condition: result-dict shape + citation ----------------------
def test_grounded_result_well_formed_and_cites_choice():
    agent = _agent(MockLLM(choice=1), ToyMemory(_two_hits(), novelty=0.1))
    result = agent.answer(_item())

    assert isinstance(result, dict)
    assert set(result) == RESULT_FIELDS
    # choice 1 -> first candidate's key is cited; not abstained.
    assert result["pred_id"] == "OMIM:1"
    assert result["abstained"] is False
    assert result["answer"] == "Disease One"
    # confidence = top Retrieved.confidence; novelty below threshold -> not flagged.
    assert result["confidence"] == 0.80
    assert result["novelty_flag"] is False
    # passthrough of the QAItem labels.
    assert result["gold_id"] == "OMIM:1" and result["answerable"] is True


def test_grounded_choice_two_cites_second_candidate():
    agent = _agent(MockLLM(choice=2), ToyMemory(_two_hits()))
    result = agent.answer(_item())
    assert result["pred_id"] == "OMIM:2"
    assert result["answer"] == "Disease Two"
    assert result["abstained"] is False


def test_grounded_dense_like_memory_same_contract():
    # A different retriever ranking (dense-RAG style) flows through identically.
    dense_like = ToyMemory(
        [Retrieved("OMIM:3", 0.77, 0.77, 1), Retrieved("OMIM:1", 0.50, 0.77, 2)],
        novelty=0.0,
    )
    result = _agent(MockLLM(choice=1), dense_like).answer(_item(gold_id="OMIM:3"))
    assert set(result) == RESULT_FIELDS
    assert result["pred_id"] == "OMIM:3"
    assert result["confidence"] == 0.77


# --- abstain paths ---------------------------------------------------------
def test_grounded_choice_zero_abstains():
    result = _agent(MockLLM(choice=0), ToyMemory(_two_hits())).answer(_item())
    assert result["abstained"] is True
    assert result["pred_id"] is None
    assert result["answer"] == "ABSTAIN"


def test_grounded_garbage_reply_falls_back_to_abstain():
    result = _agent(MockLLM(raw="I will not emit JSON"), ToyMemory(_two_hits())).answer(_item())
    assert result["abstained"] is True
    assert result["pred_id"] is None


def test_grounded_out_of_range_choice_abstains():
    # choice past the number of candidates is invalid -> safe abstain.
    result = _agent(MockLLM(choice=9), ToyMemory(_two_hits())).answer(_item())
    assert result["abstained"] is True and result["pred_id"] is None


def test_grounded_no_candidates_abstains_safely():
    # empty retrieval -> confidence 0, any choice collapses to abstain.
    result = _agent(MockLLM(choice=1), ToyMemory([], novelty=0.9)).answer(_item())
    assert result["abstained"] is True
    assert result["pred_id"] is None
    assert result["confidence"] == 0.0


# --- novelty flag honours the threshold ------------------------------------
def test_novelty_flag_above_threshold_is_true():
    agent = _agent(
        MockLLM(choice=1), ToyMemory(_two_hits(), novelty=0.9), novelty_threshold=0.5
    )
    assert agent.answer(_item(answerable=False, novel=True, gold_id="OMIM:3"))["novelty_flag"] is True


def test_novelty_flag_below_threshold_is_false():
    agent = _agent(
        MockLLM(choice=1), ToyMemory(_two_hits(), novelty=0.2), novelty_threshold=0.5
    )
    assert agent.answer(_item())["novelty_flag"] is False


# --- closed-book condition -------------------------------------------------
def test_closed_book_result_well_formed():
    result = QAAgent(MockLLM(diagnosis="Disease One"), None).answer(_item())
    assert set(result) == RESULT_FIELDS
    # closed-book: no citation, flat confidence, novelty undefined -> False.
    assert result["pred_id"] is None
    assert result["confidence"] == 0.5
    assert result["novelty_flag"] is False
    assert result["abstained"] is False
    assert result["answer"] == "Disease One"


def test_closed_book_abstain_parses():
    result = QAAgent(MockLLM(diagnosis="ABSTAIN"), None).answer(_item())
    assert result["abstained"] is True
    assert result["answer"] == "ABSTAIN"


def test_closed_book_garbage_falls_back_to_abstain():
    result = QAAgent(MockLLM(raw="totally not json"), None).answer(_item())
    assert result["abstained"] is True
    assert result["answer"] == "ABSTAIN"


def test_mockllm_never_returns_empty_and_records_calls():
    # The mock is deterministic and logs each call; it must never be the real LLM.
    llm = MockLLM(choice=1)
    _agent(llm, ToyMemory(_two_hits())).answer(_item())
    assert len(llm.calls) == 1
    assert llm.calls[0][0]["role"] == "system"


# --- driver end-to-end under --mock (no DeepSeek) --------------------------
def _load_driver():
    """Import scripts/agentic_qa.py as a module (it is a CLI, not a package)."""
    path = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "agentic_qa.py"
    spec = importlib.util.spec_from_file_location("agentic_qa_driver", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _toy_pools() -> dict:
    """A tiny stand-in for ``build_pools`` (no HPO download)."""
    from sma.eval.agentic import IndexItem

    index_items = [
        IndexItem(
            key=k,
            term_ids=KEY_TO_TERMS[k],
            text="phenotype one phenotype two",
            meta={"name": v},
        )
        for k, v in KEY_TO_NAME.items()
    ]
    answerable = [_item(gold_id="OMIM:1"), _item(gold_id="OMIM:2")]
    novel = [_item(answerable=False, novel=True, gold_id="OMIM:3")]
    return {
        "index_items": index_items,
        "answerable": answerable,
        "ook": novel,
        "novel": novel,
    }


@pytest.mark.parametrize("memory_name", ["sma", "dense", "none"])
def test_driver_runs_under_mock_and_writes_csvs(memory_name, tmp_path, monkeypatch):
    driver = _load_driver()

    # Redirect outputs to a temp dir and stub out everything that would touch the
    # real HPO files / a real retriever / the real LLM.
    monkeypatch.setattr(driver, "OUT", tmp_path)
    monkeypatch.setattr(driver, "mount", lambda graph, *a, **k: None)
    monkeypatch.setattr(driver, "load_obo", lambda path, name="": None)
    monkeypatch.setattr(driver, "build_pools", lambda *a, **k: _toy_pools())
    # SMA / dense memory constructors -> a toy memory honouring the contract.
    monkeypatch.setattr(driver, "SmaMemory", lambda mounted: ToyMemory(_two_hits(), 0.9))
    monkeypatch.setattr(driver, "DenseMemory", lambda: ToyMemory(_two_hits(), 0.1))
    # The real DeepSeek constructor must NEVER be reached on the --mock path.
    monkeypatch.setattr(
        driver,
        "make_llm",
        lambda mock: MockLLM(choice=1) if mock else _boom(),
    )

    driver.main(["--memory", memory_name, "--mock", "--pilot"])

    per_item = tmp_path / f"qa_{memory_name}.csv"
    summary = tmp_path / f"qa_{memory_name}_summary.csv"
    assert per_item.exists() and summary.exists()

    rows = list(csv.DictReader(per_item.open()))
    # answerable (2) + novel (1) = 3 per-item rows.
    assert len(rows) == 3
    assert set(RESULT_FIELDS).issubset(rows[0].keys())

    srow = next(iter(csv.DictReader(summary.open())))
    assert srow["memory"] == memory_name
    # closed-book has no retrieval -> citation cell is N/A; grounded is numeric.
    if memory_name == "none":
        assert srow["citation_faithfulness"] == "NA"
    else:
        float(srow["citation_faithfulness"])  # parses as a number


def _boom():
    raise AssertionError("real LLM constructed under --mock (must never happen)")
