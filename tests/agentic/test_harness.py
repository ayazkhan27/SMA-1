"""End-to-end test for the one-shot agentic harness on a tiny toy ontology.

A 2-memory harness (SMA + BM25) over a ~20-entity hand-built is-a ontology must
run end-to-end and return a result dict carrying both tail slices (all + rare),
per-memory AURC and novelty F1, and the SMA-vs-best paired bootstrap. No BGE /
cross-encoder is involved, so the test is fast. The harness must be
deterministic: identical seeds -> identical result.
"""

from __future__ import annotations

from sma.eval.agentic.harness import run_oneshot
from sma.eval.agentic.memories import BM25Memory, SmaMemory
from sma.ontology.graph import OntologyGraph, Term
from sma.ontology.mount import mount


def _toy_mounted() -> OntologyGraph:
    """Twelve leaves under four mid-level terms under two roots; a real is-a tree."""
    terms: dict[str, Term] = {}
    # roots
    terms["TST:R1"] = Term(id="TST:R1", name="organ system one")
    terms["TST:R2"] = Term(id="TST:R2", name="organ system two")
    # mid-level under roots
    mids = {
        "TST:M1": ("cardiac abnormality", "TST:R1"),
        "TST:M2": ("ocular abnormality", "TST:R1"),
        "TST:M3": ("hepatic abnormality", "TST:R2"),
        "TST:M4": ("renal abnormality", "TST:R2"),
    }
    for tid, (nm, parent) in mids.items():
        terms[tid] = Term(id=tid, name=nm, parents=(parent,))
    # leaves under mids
    leaves = {
        "TST:A": ("aortic murmur", "TST:M1"),
        "TST:B": ("bradycardia", "TST:M1"),
        "TST:C": ("retinal pigment", "TST:M2"),
        "TST:D": ("cataract", "TST:M2"),
        "TST:E": ("hepatic fibrosis", "TST:M3"),
        "TST:F": ("cholestasis", "TST:M3"),
        "TST:G": ("renal cyst", "TST:M4"),
        "TST:H": ("nephritis", "TST:M4"),
        "TST:I": ("septal defect", "TST:M1"),
        "TST:J": ("glaucoma", "TST:M2"),
        "TST:K": ("cirrhosis", "TST:M3"),
        "TST:L": ("renal failure", "TST:M4"),
    }
    for tid, (nm, parent) in leaves.items():
        terms[tid] = Term(id=tid, name=nm, parents=(parent,))
    return OntologyGraph(name="toy", version="v0", terms=terms)


def _toy_records() -> dict[str, set[str]]:
    """Twenty entities, each a small set of leaf terms (some overlapping)."""
    leaf_ids = [
        "TST:A", "TST:B", "TST:C", "TST:D", "TST:E", "TST:F",
        "TST:G", "TST:H", "TST:I", "TST:J", "TST:K", "TST:L",
    ]
    records: dict[str, set[str]] = {}
    import random

    rng = random.Random(0)
    for i in range(20):
        k = rng.randint(3, 5)
        records[f"ent_{i:02d}"] = set(rng.sample(leaf_ids, k))
    return records


def _run(seeds=(7, 17, 23)):
    mounted = mount(_toy_mounted())
    records = _toy_records()
    memories = [SmaMemory(mounted), BM25Memory()]
    return run_oneshot(
        "toy",
        mounted,
        records,
        memories,
        seeds=seeds,
        n_index=20,
        n_query=12,
        holdout_frac=0.1,
    )


def test_harness_runs_end_to_end():
    result = _run()

    assert result["arm"] == "toy"
    assert result["memories"] == ["sma", "bm25"]
    assert result["n_all"] > 0
    assert result["n_novel"] > 0

    # per-memory: tail slices (all + rare), AURC, novelty F1.
    for m in ("sma", "bm25"):
        pm = result["per_memory"][m]
        for k in ("top1", "top5", "top10"):
            assert set(pm["tail"][k]) == {"all", "rare"}
            assert 0.0 <= pm["tail"][k]["all"] <= 1.0
            assert 0.0 <= pm["tail"][k]["rare"] <= 1.0
        assert pm["aurc"] >= 0.0
        assert 0.0 <= pm["novelty_f1"] <= 1.0

    # primary: SMA vs best enterprise (here BM25) bootstrap.
    primary = result["primary"]
    assert primary is not None
    assert primary["a"] == "sma"
    assert primary["best_enterprise"] == "bm25"
    for key in ("delta_top5", "ci_low", "ci_high", "p_value", "cliffs"):
        assert key in primary
    assert -1.0 <= primary["cliffs"] <= 1.0


def test_harness_is_deterministic():
    a = _run()
    b = _run()
    assert a == b


def test_harness_changes_with_seed():
    """Different seeds may change the numbers; structure must stay intact."""
    a = _run(seeds=(7,))
    b = _run(seeds=(99,))
    assert set(a["per_memory"]) == set(b["per_memory"])
    assert a["arm"] == b["arm"]
