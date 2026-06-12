"""Unit tests for sma.eval.stats (prereg section 5 statistics)."""

from __future__ import annotations

import pytest

from sma.eval.stats import cliffs_delta, holm_bonferroni, paired_bootstrap


# ---------------------------------------------------------------------------
# paired_bootstrap
# ---------------------------------------------------------------------------


def test_paired_bootstrap_ci_excludes_zero_for_obvious_difference():
    a = [1.0] * 20
    b = [0.0] * 20
    out = paired_bootstrap(a, b, n_resamples=2000, seed=12345)
    assert out["delta"] == pytest.approx(1.0)
    assert out["ci_low"] > 0.0
    assert out["ci_high"] >= out["ci_low"]
    assert out["p_value"] < 0.05


def test_paired_bootstrap_ci_includes_zero_for_identical_lists():
    a = [0.3, 0.7, 0.1, 0.9, 0.5]
    out = paired_bootstrap(a, list(a), n_resamples=2000, seed=12345)
    assert out["delta"] == pytest.approx(0.0)
    assert out["ci_low"] <= 0.0 <= out["ci_high"]
    assert out["p_value"] == pytest.approx(1.0)


def test_paired_bootstrap_noisy_but_consistent_advantage():
    # Every pair favors a by 0.5; CI must exclude zero in the + direction.
    b = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.1, 0.3, 0.5, 0.7] * 3
    a = [x + 0.5 for x in b]
    out = paired_bootstrap(a, b, n_resamples=2000, seed=12345)
    assert out["delta"] == pytest.approx(0.5)
    assert out["ci_low"] > 0.0
    assert out["p_value"] < 0.05


def test_paired_bootstrap_deterministic_given_seed():
    a = [0.1, 0.5, 0.9, 0.2, 0.8, 0.4]
    b = [0.2, 0.3, 0.7, 0.4, 0.5, 0.6]
    one = paired_bootstrap(a, b, n_resamples=500, seed=777)
    two = paired_bootstrap(a, b, n_resamples=500, seed=777)
    assert one == two
    other_seed = paired_bootstrap(a, b, n_resamples=500, seed=778)
    assert other_seed["delta"] == one["delta"]  # observed delta is seed-free


def test_paired_bootstrap_sign_symmetry():
    a = [1.0, 0.9, 0.8, 1.0, 0.7]
    b = [0.1, 0.2, 0.0, 0.3, 0.1]
    fwd = paired_bootstrap(a, b, n_resamples=1000, seed=12345)
    rev = paired_bootstrap(b, a, n_resamples=1000, seed=12345)
    assert rev["delta"] == pytest.approx(-fwd["delta"])
    assert rev["ci_high"] < 0.0


def test_paired_bootstrap_rejects_bad_input():
    with pytest.raises(ValueError):
        paired_bootstrap([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        paired_bootstrap([], [])


# ---------------------------------------------------------------------------
# holm_bonferroni
# ---------------------------------------------------------------------------


def test_holm_known_three_comparison_case():
    raw = {"x": 0.01, "z": 0.03, "y": 0.04}
    adj = holm_bonferroni(raw)
    # sorted: x (0.01*3=0.03), z (0.03*2=0.06), y (max(0.04*1, 0.06)=0.06)
    assert adj["x"] == pytest.approx(0.03)
    assert adj["z"] == pytest.approx(0.06)
    assert adj["y"] == pytest.approx(0.06)


def test_holm_preserves_order_and_monotonicity():
    raw = {"a": 0.001, "b": 0.20, "c": 0.04, "d": 0.012}
    adj = holm_bonferroni(raw)
    ordered = sorted(raw, key=raw.__getitem__)
    adj_in_order = [adj[k] for k in ordered]
    assert adj_in_order == sorted(adj_in_order)  # step-down never decreases
    for key in raw:
        assert adj[key] >= raw[key]


def test_holm_caps_at_one_and_single_comparison_identity():
    assert holm_bonferroni({"only": 0.04}) == {"only": pytest.approx(0.04)}
    adj = holm_bonferroni({"a": 0.6, "b": 0.9})
    assert adj["a"] == pytest.approx(1.0)  # 0.6 * 2 capped
    assert adj["b"] == pytest.approx(1.0)


def test_holm_rejects_out_of_range_p():
    with pytest.raises(ValueError):
        holm_bonferroni({"bad": 1.5})


# ---------------------------------------------------------------------------
# cliffs_delta
# ---------------------------------------------------------------------------


def test_cliffs_delta_fully_separated():
    assert cliffs_delta([2.0, 3.0, 4.0], [0.0, 1.0]) == pytest.approx(1.0)
    assert cliffs_delta([0.0, 1.0], [2.0, 3.0, 4.0]) == pytest.approx(-1.0)


def test_cliffs_delta_identical_lists_is_zero():
    assert cliffs_delta([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0)


def test_cliffs_delta_known_partial_overlap():
    # pairs: (1,2)<, (1,4)<, (3,2)>, (3,4)< -> (1 - 3) / 4 = -0.5
    assert cliffs_delta([1.0, 3.0], [2.0, 4.0]) == pytest.approx(-0.5)


def test_cliffs_delta_ties_do_not_count():
    # pairs vs [1,1]: (1,1)=, (1,1)=, (2,1)>, (2,1)> -> 2/4
    assert cliffs_delta([1.0, 2.0], [1.0, 1.0]) == pytest.approx(0.5)


def test_cliffs_delta_rejects_empty():
    with pytest.raises(ValueError):
        cliffs_delta([], [1.0])
