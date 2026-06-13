"""Small-N regression for the Medicine arm (HPO rare-disease).

Loads the real HPO ontology + phenotype records, runs the one-shot harness on a
small slice (n_index=200, n_query=20, SMA + BM25 only, one seed) and asserts the
result dict is well-formed and SMA's answerable tail top-5 is at least BM25's --
the de-risk direction (SMA should not lose to lexical BM25 on the tail). No BGE /
cross-encoder is loaded, so this stays fast (~tens of seconds).
"""

from __future__ import annotations

import pathlib

import pytest

from sma.eval.agentic import BM25Memory, SmaMemory, run_oneshot
from sma.eval.agentic.arms import medicine

_DATA = pathlib.Path(medicine.HP_OBO)


@pytest.mark.skipif(not _DATA.exists(), reason="HPO data not present")
def test_medicine_arm_small_n_regression():
    mounted, records = medicine.load()
    assert len(records) > 200  # enough diseases to index a 200-entity slice

    memories = [SmaMemory(mounted), BM25Memory()]
    result = run_oneshot(
        "medicine",
        mounted,
        records,
        memories,
        seeds=(7,),
        n_index=200,
        n_query=20,
        holdout_frac=0.1,
    )

    # well-formed result dict
    assert result["arm"] == "medicine"
    assert result["memories"] == ["sma", "bm25"]
    assert result["n_all"] > 0
    assert result["n_novel"] > 0
    for m in ("sma", "bm25"):
        pm = result["per_memory"][m]
        for k in ("top1", "top5", "top10"):
            assert set(pm["tail"][k]) == {"all", "rare"}
            assert 0.0 <= pm["tail"][k]["all"] <= 1.0
            assert 0.0 <= pm["tail"][k]["rare"] <= 1.0
        assert pm["aurc"] >= 0.0
        assert 0.0 <= pm["novelty_f1"] <= 1.0

    primary = result["primary"]
    assert primary is not None
    assert primary["a"] == "sma"
    assert primary["best_enterprise"] == "bm25"

    # de-risk direction: SMA's answerable tail top-5 >= BM25's (sanity).
    sma_t5 = result["per_memory"]["sma"]["tail"]["top5"]["all"]
    bm25_t5 = result["per_memory"]["bm25"]["tail"]["top5"]["all"]
    assert sma_t5 >= bm25_t5
