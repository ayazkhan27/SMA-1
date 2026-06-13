"""Tests for the Phase 5 LLM-QA trustworthy-QA metrics.

Hand-built result lists with known expected values. Covers the four axes plus
the registered edge cases: closed-book citation N/A, the answerable/ook/novel
partition, and the risk-coverage AURC wiring (confidence = 1 - abstain_flag).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from sma.eval.agentic.metrics import risk_coverage_aurc
from sma.eval.agentic_qa.metrics import (
    abstention,
    accuracy,
    citation_faithfulness,
    novelty_recall,
)


@dataclass
class R:
    """A minimal agent-result object (mirrors the dict result fields)."""

    gold_id: str = "G"
    gold_name: str = "gold"
    answerable: bool = True
    novel: bool = False
    abstained: bool = False
    pred_id: str | None = None
    answer: str = ""
    novelty_flag: bool = False
    confidence: float = 1.0


# --- accuracy --------------------------------------------------------------
def test_accuracy_grounded_id_match():
    # answerable only; grounded condition => correctness is pred_id == gold_id.
    results = [
        R(gold_id="A", answerable=True, abstained=False, pred_id="A"),  # hit
        R(gold_id="B", answerable=True, abstained=False, pred_id="X"),  # wrong id
        R(gold_id="C", answerable=True, abstained=True, pred_id="C"),   # abstained
        R(gold_id="D", answerable=False, abstained=False, pred_id="D"),  # ook: ignored
    ]
    # 1 hit over 3 answerable.
    assert accuracy(results) == 1 / 3


def test_accuracy_closed_book_name_match():
    # closed-book: pred_id is None -> case-insensitive substring name-match.
    results = [
        R(gold_name="Marfan syndrome", pred_id=None, answer="Marfan Syndrome"),  # ci hit
        R(gold_name="Cystic fibrosis", pred_id=None, answer="I think cystic fibrosis."),  # substring hit
        R(gold_name="Fabry disease", pred_id=None, answer="Pompe disease"),  # miss
        R(gold_name="Gaucher", pred_id=None, answer=""),  # empty answer -> miss
    ]
    assert accuracy(results) == 2 / 4


def test_accuracy_no_answerable_is_zero():
    results = [R(answerable=False), R(answerable=False, novel=True)]
    assert accuracy(results) == 0.0


# --- citation faithfulness -------------------------------------------------
def test_citation_faithfulness_over_answered_cited():
    results = [
        R(gold_id="A", answerable=True, abstained=False, pred_id="A"),  # faithful
        R(gold_id="B", answerable=True, abstained=False, pred_id="Z"),  # cited != gold
        R(gold_id="C", answerable=True, abstained=True, pred_id="C"),   # abstained: skip
        R(gold_id="D", answerable=True, abstained=False, pred_id=None),  # no citation: skip
        R(gold_id="E", answerable=False, abstained=False, pred_id="E"),  # ook: skip
    ]
    # applicable = {A, B}; faithful = {A} -> 1/2.
    assert citation_faithfulness(results) == 1 / 2


def test_citation_faithfulness_na_for_closed_book():
    # closed-book condition: nobody cites -> N/A (None), NOT 0.0.
    results = [
        R(answerable=True, abstained=False, pred_id=None, answer="x"),
        R(answerable=True, abstained=True, pred_id=None),
    ]
    assert citation_faithfulness(results) is None


# --- abstention ------------------------------------------------------------
def _abstention_fixture() -> list[R]:
    return [
        R(gold_id="A1", answerable=True, abstained=False, pred_id="A1"),  # answered correct
        R(gold_id="A2", answerable=True, abstained=False, pred_id="WW"),  # answered wrong
        R(gold_id="A3", answerable=True, abstained=True, pred_id=None),   # false abstain
        R(gold_id="O1", answerable=False, novel=False, abstained=True),   # correct abstain
        R(gold_id="O2", answerable=False, novel=False, abstained=False, pred_id="ZZ"),  # hallucinated
    ]


def test_abstention_fractions():
    out = abstention(_abstention_fixture())
    # ook = {O1, O2}; abstained ook = {O1} -> recall 1/2.
    assert out["abstain_recall"] == 1 / 2
    # answerable = {A1,A2,A3}; wrongly abstained = {A3} -> 1/3.
    assert math.isclose(out["false_abstain"], 1 / 3)
    # union of 5; selective-correct = {A1 answered-right, O1 abstained-right} -> 2/5.
    assert out["selective_accuracy"] == 2 / 5


def test_abstention_aurc_matches_direct_call():
    # confidence = 1 - abstain_flag; correct = answered & right.
    # union order: A1,A2,A3 (answerable) then O1,O2 (ook).
    out = abstention(_abstention_fixture())
    confidences = [1.0, 1.0, 0.0, 0.0, 1.0]  # A1,A2,A3,O1,O2
    correct = [True, False, False, False, False]
    exp_aurc, exp_pts = risk_coverage_aurc(confidences, correct)
    assert math.isclose(out["aurc"], exp_aurc)
    assert out["rc_points"] == exp_pts
    # spot-check the hand-computed AURC = mean(0, .5, 2/3, .75, .8).
    assert math.isclose(out["aurc"], (0 + 0.5 + 2 / 3 + 0.75 + 0.8) / 5)


def test_abstention_novel_excluded_from_union():
    # A novel item is neither answerable nor ook -> it must not move any axis.
    base = _abstention_fixture()
    with_novel = base + [
        R(gold_id="N1", answerable=False, novel=True, abstained=False, novelty_flag=True),
    ]
    assert abstention(with_novel) == abstention(base)


def test_abstention_empty_pools_no_zero_division():
    out = abstention([])
    assert out["abstain_recall"] == 0.0
    assert out["false_abstain"] == 0.0
    assert out["selective_accuracy"] == 0.0
    assert out["aurc"] == 0.0
    assert out["rc_points"] == []


def test_abstention_perfect_selective_agent():
    # answers every answerable correctly, abstains on every ook.
    results = [
        R(gold_id="A", answerable=True, abstained=False, pred_id="A"),
        R(gold_id="B", answerable=True, abstained=False, pred_id="B"),
        R(gold_id="O", answerable=False, novel=False, abstained=True),
    ]
    out = abstention(results)
    assert out["abstain_recall"] == 1.0
    assert out["false_abstain"] == 0.0
    assert out["selective_accuracy"] == 1.0
    # Coverage order A,B (correct, conf 1) then O (abstain, conf 0, never
    # "correct" for the risk curve): risk stays 0 across the answered head, then
    # the trailing abstain is forced into coverage -> AURC = mean(0, 0, 1/3).
    assert math.isclose(out["aurc"], 1 / 9)


# --- novelty recall --------------------------------------------------------
def test_novelty_recall_over_novel_only():
    results = [
        R(novel=True, novelty_flag=True),    # flagged
        R(novel=True, novelty_flag=False),   # missed
        R(novel=True, novelty_flag=True),    # flagged
        R(novel=False, novelty_flag=True),   # not novel: ignored (would be a false flag)
        R(answerable=True, novelty_flag=False),
    ]
    assert novelty_recall(results) == 2 / 3


def test_novelty_recall_no_novel_is_zero():
    assert novelty_recall([R(novel=False), R(answerable=True)]) == 0.0


# --- dict inputs work identically to objects -------------------------------
def test_metrics_accept_plain_dicts():
    results = [
        {"gold_id": "A", "gold_name": "a", "answerable": True, "novel": False,
         "abstained": False, "pred_id": "A", "answer": "", "novelty_flag": False,
         "confidence": 1.0},
        {"gold_id": "B", "gold_name": "b", "answerable": True, "novel": False,
         "abstained": True, "pred_id": None, "answer": "", "novelty_flag": False,
         "confidence": 0.0},
        {"gold_id": "O", "gold_name": "o", "answerable": False, "novel": False,
         "abstained": True, "pred_id": None, "answer": "", "novelty_flag": False,
         "confidence": 0.0},
        {"gold_id": "N", "gold_name": "n", "answerable": False, "novel": True,
         "abstained": False, "pred_id": None, "answer": "", "novelty_flag": True,
         "confidence": 0.0},
    ]
    assert accuracy(results) == 1 / 2          # A hit, B abstained, over 2 answerable
    assert citation_faithfulness(results) == 1.0  # only A is answered+cited, faithful
    assert novelty_recall(results) == 1.0      # the one novel item is flagged
    out = abstention(results)
    assert out["abstain_recall"] == 1.0        # the one ook item abstained
    assert math.isclose(out["false_abstain"], 1 / 2)  # B wrongly abstained
