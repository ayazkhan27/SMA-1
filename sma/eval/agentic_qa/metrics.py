"""Trustworthy-QA metrics for the Phase 5 LLM-QA harness (prereg v2 section 4).

Given per-item agent results, compute the four pre-registered axes that
distinguish a *verifiable specialist* from a confident-but-opaque RAG agent:

* :func:`accuracy` — answer correct on the **answerable** pool (the accuracy
  floor; the capability gains must not cost accuracy).
* :func:`citation_faithfulness` — ALCE-style support score over **answered
  answerable** items: did the cited candidate actually turn out to be the gold?
  N/A (``None``) for the closed-book condition, which has no citation.
* :func:`abstention` — selective prediction over the union of **answerable**
  (should answer) and **held-out / out-of-knowledge** (should abstain):
  abstain-recall, false-abstain, selective-accuracy, plus the risk-coverage AURC
  with confidence ``= 1 - abstain_flag``.
* :func:`grounding_auroc` — threshold-free discrimination of the RAW grounding
  score: AUROC for separating answerable (high score) from held-out (low score).
  The intrinsic "can the memory tell known from unknown" signal, independent of
  where the abstention threshold sits.
* :func:`novelty_recall` / :func:`novelty_f1` — recall (and precision/F1 against
  answerable false-alarms) of the novelty flag over the **novel** pool.

A result is a simple dict or object exposing: ``gold_id``, ``gold_name``,
``answerable``, ``novel``, ``abstained`` (bool), ``pred_id`` (str | None),
``answer`` (str), ``novelty_flag`` (bool), ``confidence`` (float),
``grounding_score`` (float | None). The data have two disjoint groups:
**answerable** (the gold disease IS indexed) and **held-out** (the gold disease
is NOT indexed). A held-out case is simultaneously out-of-knowledge (the agent
should ABSTAIN) and novel (the agent should FLAG it) — both correct trustworthy
behaviours on the same unindexed case — so abstention and novelty are scored on
the same held-out items (``answerable == False``, ``novel == True``).
"""

from __future__ import annotations

from typing import Any

from sma.eval.agentic.metrics import risk_coverage_aurc


def _get(result: Any, field: str, default: Any = None) -> Any:
    """Read ``field`` from a result whether it is a dict or an object."""
    if isinstance(result, dict):
        return result.get(field, default)
    return getattr(result, field, default)


def _correct(result: Any) -> bool:
    """Did the agent name the right entity? (grounded id-match, else name-match).

    If the agent cited a candidate (``pred_id`` is not None), correctness is an
    exact id match against the gold. For the closed-book condition (no
    retrieval, ``pred_id`` is None) we fall back to a case-insensitive substring
    name-match of the free-text ``answer`` against ``gold_name``.
    """
    pred_id = _get(result, "pred_id")
    if pred_id is not None:
        return pred_id == _get(result, "gold_id")
    answer = (_get(result, "answer") or "").strip().lower()
    gold_name = (_get(result, "gold_name") or "").strip().lower()
    if not answer or not gold_name:
        return False
    return gold_name in answer or answer in gold_name


def accuracy(results: list[Any]) -> float:
    """Fraction of **answerable** items answered (not abstained) and correct.

    Returns 0.0 when there are no answerable items (no division by zero).
    """
    answerable = [r for r in results if _get(r, "answerable")]
    if not answerable:
        return 0.0
    hits = sum(
        1 for r in answerable if not _get(r, "abstained") and _correct(r)
    )
    return hits / len(answerable)


def citation_faithfulness(results: list[Any]) -> float | None:
    """Support score over **answered answerable** items with a citation.

    Over answerable items that were answered (not abstained) *and* carry a
    citation (``pred_id`` is not None), the fraction whose cited candidate is in
    fact the gold (``pred_id == gold_id``). Items with no retrieval/citation are
    skipped. Returns ``None`` (N/A) when no item is applicable — e.g. the
    closed-book condition, where citation-faithfulness is undefined.
    """
    cited = [
        r
        for r in results
        if _get(r, "answerable")
        and not _get(r, "abstained")
        and _get(r, "pred_id") is not None
    ]
    if not cited:
        return None
    hits = sum(1 for r in cited if _get(r, "pred_id") == _get(r, "gold_id"))
    return hits / len(cited)


def abstention(results: list[Any]) -> dict[str, Any]:
    """Selective prediction over {answerable should-answer} + {held-out should-abstain}.

    The should-abstain (out-of-knowledge) set is every **held-out** item — i.e.
    ``not answerable`` — because an unindexed disease is out-of-knowledge whether
    or not it is also flagged novel (it always is, here). Returns a dict with:

    * ``abstain_recall`` — fraction of ook items that abstained;
    * ``false_abstain`` — fraction of answerable items that wrongly abstained;
    * ``selective_accuracy`` — over the answerable+ook union, the fraction that
      either answered correctly (answerable) or correctly abstained (ook);
    * ``aurc`` / ``rc_points`` — risk-coverage curve over the same union with
      ``confidence = 1 - abstain_flag`` and ``correct = answered & correct``
      (an abstain is never "correct" for the risk curve; a wrong answer is the
      worst case, surfaced first at high confidence).

    Empty pools yield 0.0 for their respective fractions (no division by zero).
    """
    answerable = [r for r in results if _get(r, "answerable")]
    ook = [r for r in results if not _get(r, "answerable")]

    n_ook_abstain = sum(1 for r in ook if _get(r, "abstained"))
    abstain_recall = n_ook_abstain / len(ook) if ook else 0.0

    n_ans_abstain = sum(1 for r in answerable if _get(r, "abstained"))
    false_abstain = n_ans_abstain / len(answerable) if answerable else 0.0

    union = answerable + ook
    n_selective_ok = 0
    confidences: list[float] = []
    correct: list[bool] = []
    for r in union:
        abstained = bool(_get(r, "abstained"))
        answerable_r = bool(_get(r, "answerable"))
        answered_correct = (not abstained) and _correct(r)
        # selective-accuracy: answer right (answerable) OR abstain right (ook).
        if answerable_r:
            if answered_correct:
                n_selective_ok += 1
        else:  # ook -> the right move is to abstain
            if abstained:
                n_selective_ok += 1
        # risk-coverage: coverage = answered, correctness = answered & right.
        confidences.append(0.0 if abstained else 1.0)
        correct.append(answered_correct)

    selective_accuracy = n_selective_ok / len(union) if union else 0.0
    aurc, rc_points = risk_coverage_aurc(confidences, correct)

    return {
        "abstain_recall": abstain_recall,
        "false_abstain": false_abstain,
        "selective_accuracy": selective_accuracy,
        "aurc": aurc,
        "rc_points": rc_points,
    }


def novelty_recall(results: list[Any]) -> float:
    """Fraction of **novel** items the agent flagged (``novelty_flag`` True).

    Returns 0.0 when there are no novel items (no division by zero).
    """
    novel = [r for r in results if _get(r, "novel")]
    if not novel:
        return 0.0
    hits = sum(1 for r in novel if _get(r, "novelty_flag"))
    return hits / len(novel)


def novelty_f1(results: list[Any]) -> dict[str, float]:
    """Precision / recall / F1 of the novelty flag (held-out positive, answerable negative).

    Treats **novel** (held-out) items as positives and **answerable** (indexed)
    items as negatives, so a novelty flag fired on an answerable case counts as a
    false alarm. This penalises an agent that flags everything (recall 1.0 is
    cheap; F1 is not). Returns ``precision`` / ``recall`` / ``f1`` / ``fpr``
    (false-positive rate on answerable). Empty-pool fractions collapse to 0.0.
    """
    novel = [r for r in results if _get(r, "novel")]
    answerable = [r for r in results if _get(r, "answerable")]

    tp = sum(1 for r in novel if _get(r, "novelty_flag"))
    fn = len(novel) - tp
    fp = sum(1 for r in answerable if _get(r, "novelty_flag"))

    recall = tp / len(novel) if novel else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    fpr = fp / len(answerable) if answerable else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "fpr": fpr}


def auroc(pos: list[float], neg: list[float]) -> float:
    """Rank-based (Mann-Whitney U) AUROC that ``pos`` scores exceed ``neg`` scores.

    Concordant pairs count 1, ties 0.5. ``pos`` are the should-answer (high-score)
    class, ``neg`` the should-abstain (low-score) class. Assumes both non-empty.
    Reused by the driver to score the calibration split and by
    :func:`grounding_auroc` to score the test split.
    """
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def grounding_auroc(results: list[Any]) -> float | None:
    """Threshold-free discrimination of the RAW grounding score: answerable vs held-out.

    AUROC that the top structural grounding score is higher on **answerable**
    (the gold disease is indexed -> should ground) than on **held-out** items
    (not indexed -> should not ground). This is the intrinsic known-vs-unknown
    signal, independent of where any abstention threshold is set. Returns ``None``
    (N/A) when either group is empty or carries no grounding score (closed-book).
    """
    pos = [
        s
        for r in results
        if _get(r, "answerable")
        and (s := _get(r, "grounding_score")) is not None
    ]
    neg = [
        s
        for r in results
        if not _get(r, "answerable")
        and (s := _get(r, "grounding_score")) is not None
    ]
    if not pos or not neg:
        return None
    return auroc(pos, neg)
