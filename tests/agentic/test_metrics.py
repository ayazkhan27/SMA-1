"""Tests for the agentic suite metrics (Task 5)."""

from __future__ import annotations

import math

from sma.eval.agentic.metrics import (
    ABSENT_RANK,
    novelty_f1,
    risk_coverage_aurc,
    tail_topk,
)


def test_tail_topk_toy_rows():
    rows = [
        {"sma": 1, "bm25": 3, "rare": False},
        {"sma": 2, "bm25": ABSENT_RANK, "rare": True},
        {"sma": 6, "bm25": 1, "rare": True},
        {"sma": 1, "bm25": 4, "rare": False},
    ]
    out = tail_topk(rows, k=5)

    # ALL slice: sma ranks <=5 are rows 0,1,3 => 3/4; bm25 are rows 0,2,3 => 3/4
    assert out["sma"]["all"] == 3 / 4
    assert out["bm25"]["all"] == 3 / 4

    # RARE slice (rows 1,2): sma row1 rank 2 (<=5), row2 rank 6 (>5) => 1/2;
    # bm25 row1 absent (>5), row2 rank 1 (<=5) => 1/2
    assert out["sma"]["rare"] == 1 / 2
    assert out["bm25"]["rare"] == 1 / 2

    # methods discovered from rows, "rare" excluded
    assert set(out) == {"sma", "bm25"}


def test_tail_topk_empty_rare_slice():
    rows = [{"sma": 1, "rare": False}, {"sma": 2, "rare": False}]
    out = tail_topk(rows, k=1)
    assert out["sma"]["all"] == 1 / 2
    assert out["sma"]["rare"] == 0.0  # no rare rows -> 0, no ZeroDivisionError


def test_aurc_calibrated_beats_miscalibrated():
    # 5 correct, 5 wrong. Perfect calibration: every correct item is more
    # confident than every wrong one.
    correct = [True, True, True, True, True, False, False, False, False, False]
    calibrated_conf = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    # Miscalibrated (reversed): the wrong items are the most confident.
    miscalibrated_conf = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    aurc_good, pts_good = risk_coverage_aurc(calibrated_conf, correct)
    aurc_bad, _ = risk_coverage_aurc(miscalibrated_conf, correct)

    assert aurc_good < aurc_bad
    # Perfectly calibrated: head is all-correct, so risk stays 0 until the
    # first error surfaces (at 60% coverage here).
    assert pts_good[0] == (0.1, 0.0)
    assert pts_good[4] == (0.5, 0.0)
    # full curve always ends at full coverage with the overall error rate
    cov_end, risk_end = pts_good[-1]
    assert cov_end == 1.0
    assert math.isclose(risk_end, 0.5)


def test_aurc_perfect_ranker_is_lower_than_random():
    correct = [True, True, True, True, False, False, False, False]
    perfect = [0.99, 0.98, 0.97, 0.96, 0.04, 0.03, 0.02, 0.01]
    # "random" / uninformative: ties everywhere -> stable sort keeps order,
    # interleaving correct and wrong.
    flat_conf = [0.5] * 8
    flat_correct = [True, False, True, False, True, False, True, False]

    aurc_perfect, pts = risk_coverage_aurc(perfect, correct)
    aurc_random, _ = risk_coverage_aurc(flat_conf, flat_correct)
    assert aurc_perfect < aurc_random
    # A perfect ranker keeps risk at 0 across the all-correct head; risk only
    # rises once the wrong items are forced into coverage.
    assert pts[3] == (0.5, 0.0)
    assert aurc_perfect > 0.0


def test_aurc_empty():
    assert risk_coverage_aurc([], []) == (0.0, [])


def test_novelty_f1_hand_computed():
    # tp=2 (idx 0,4), fp=1 (idx 1), fn=1 (idx 3) -> prec=rec=2/3 -> F1=2/3
    pred = [True, True, False, False, True]
    truth = [True, False, False, True, True]
    assert math.isclose(novelty_f1(pred, truth), 2 / 3)


def test_novelty_f1_perfect_and_degenerate():
    assert novelty_f1([True, False, True], [True, False, True]) == 1.0
    # no predictions and no truth -> 0.0 (no division by zero)
    assert novelty_f1([False, False], [False, False]) == 0.0
    # predicts none novel but some are -> recall 0 -> F1 0
    assert novelty_f1([False, False], [True, False]) == 0.0
