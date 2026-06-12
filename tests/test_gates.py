from __future__ import annotations

import json

import pytest

from sma.agent.policies import reject_free_text_facts
from sma.agent.comparison import ComparisonFramework, demo_corpus
from sma.agent.llm import LocalOrchestrator
from sma.agent.service import MemoryService
from sma.encoders import get_encoder
from sma.eval.arn import arn_choice_corpus, validate_columns
from sma.eval.report import CSV_SCHEMAS, run_fixture_eval
from sma.eval.ssb_generator import generate_triples
from sma.index.content_vectors import cosine, functor_vector
from sma.index.macfac import MacFacIndex
from sma.ir.schema import entity, make_case, stmt
from sma.ir.sexpr import canonical_case_text, loads_case
from sma.match import candidate_inferences, match_cases, verify_inference
from sma.match.conflicts import kernels_conflict, structurally_consistent
from sma.match.kernels import build_kernels
from sma.sage import SagePool, support_probability
from sma.store import CaseStore


def water_heat_cases():
    pipe = entity("pipe", "system")
    water = entity("water", "fluid")
    heat = entity("heat", "energy")
    base_pressure = stmt("pressure", pipe, water)
    base_flow = stmt("waterFlow", water, pipe)
    target_temp = stmt("temperature", pipe, heat)
    target_flow = stmt("heatFlow", heat, pipe)
    base = make_case(
        [
            base_pressure,
            base_flow,
            stmt("cause", base_pressure, base_flow),
            stmt("viscosity", water),
        ],
        {"fixture": "water"},
    )
    target = make_case(
        [
            target_temp,
            target_flow,
            stmt("cause", target_temp, target_flow),
        ],
        {"fixture": "heat"},
    )
    return base, target


def solar_atom_cases():
    sun = entity("sun", "body")
    planet = entity("planet", "body")
    nucleus = entity("nucleus", "body")
    electron = entity("electron", "body")
    base_attract = stmt("attractsGravity", sun, planet)
    target_attract = stmt("attractsElectrostatic", nucleus, electron)
    base = make_case([base_attract, stmt("causes", base_attract, stmt("orbit", planet, sun))])
    target = make_case([target_attract, stmt("causes", target_attract, stmt("orbit", electron, nucleus))])
    return base, target


@pytest.mark.gate_G0
def test_bootstrap_materials_exist():
    assert json.loads(open("data/manifests/datasets.json", encoding="utf-8").read())["arn"]["doi"] == "10.5281/zenodo.11044026"
    assert "retrieval_runs.csv" in CSV_SCHEMAS
    assert "SMA-MVP-1" in open("GOALS.md", encoding="utf-8").read()


@pytest.mark.gate_G1
def test_ir_roundtrip_and_store(tmp_path):
    case = make_case([stmt("cause", stmt("timeout", "svc", "db"), stmt("retry", "svc", "db"))])
    text = canonical_case_text(case.statements)
    roundtrip = make_case(loads_case(text))
    assert canonical_case_text(roundtrip.statements) == text
    store = CaseStore(tmp_path)
    store.put(case)
    loaded = store.get(case.case_id)
    assert canonical_case_text(loaded.statements) == text
    assert store.replay_wal() == [case.case_id]


@pytest.mark.gate_G2
def test_matcher_canonical_battery_and_invariants():
    base, target = water_heat_cases()
    kernels = build_kernels(base, target)
    assert kernels
    assert all(structurally_consistent(kernel.hypotheses) for kernel in kernels)
    for i, left in enumerate(kernels):
        for right in kernels[i + 1 :]:
            union_ok = structurally_consistent(left.hypotheses + right.hypotheses)
            assert union_ok is (not kernels_conflict(left, right))
    gmap = match_cases(base, target)
    rows = {(c["base"], c["target"]) for c in gmap.correspondences}
    assert any("pressure" in b and "temperature" in t for b, t in rows)
    assert any("waterFlow" in b and "heatFlow" in t for b, t in rows)
    inferences = candidate_inferences(gmap)
    assert any("viscosity" in inference.inference_sexpr for inference in inferences)

    solar, atom = solar_atom_cases()
    gmap2 = match_cases(solar, atom)
    rows2 = {(c["base"], c["target"]) for c in gmap2.correspondences}
    assert any("sun" in b and "nucleus" in t for b, t in rows2)


@pytest.mark.gate_G3
def test_encoders_are_deterministic():
    log = "INFO DataNode blk_123 timeout to 10.0.0.1\nWARN DataNode retry blk_123 failed"
    first = get_encoder("logs").encode(log).case
    second = get_encoder("logs").encode(log).case
    assert canonical_case_text(first.statements) == canonical_case_text(second.statements)
    assert first.case_id == second.case_id
    assert any(s.functor == "cause" for s in first.statements)

    code = "import os\n\ndef f(x):\n    return os.path.join('a', x)\n"
    code_case = get_encoder("code").encode(code).case
    assert any(s.functor == "defines" for s in code_case.statements)
    assert any(s.functor == "calls" for s in code_case.statements)

    prose = "A timeout happened because the database was saturated."
    prose_first = get_encoder("prose_tier1").encode(prose).case
    prose_second = get_encoder("prose_tier1").encode(prose).case
    assert canonical_case_text(prose_first.statements) == canonical_case_text(prose_second.statements)


@pytest.mark.gate_G0
def test_arn_helper_matches_downloaded_schema(tmp_path):
    columns = [
        "id",
        "proverb",
        "query_narrative",
        "first_choice",
        "second_choice",
        "distractor_similarity",
        "analogy_level",
        "correct_answer",
    ]
    assert validate_columns(columns)
    path = tmp_path / "arn.csv"
    path.write_text(
        ",".join(columns)
        + "\n"
        + "1,Count your blessings,Query story,Correct story,Distractor story,high,far,1\n",
        encoding="utf-8",
    )
    corpus, query = arn_choice_corpus(path, limit=1)
    assert "Correct story" in corpus
    assert "label=correct" in corpus
    assert query == "Query story"


@pytest.mark.gate_G4
def test_macfac_certified_matches_bruteforce():
    base, target = water_heat_cases()
    solar, atom = solar_atom_cases()
    index = MacFacIndex()
    index.build([base, solar])
    retrieved = index.retrieve(target, k=2, shortlist=2)
    brute = index.brute_force(target, k=2)
    assert [r.case_id for r in retrieved] == [r.case_id for r in brute]
    assert retrieved[0].certified


@pytest.mark.gate_G4
def test_macfac_lattice_bridges_disjoint_vocabularies():
    """De-circularized SSB: query and analog share ZERO surface vocabulary;
    the declared lattice is the only bridge (no string tricks)."""
    from sma.eval.ssb_generator import build_canonicalizer
    from sma.eval.ssb_eval import ssb_config

    triple = generate_triples(1, seed=11)[0]
    canon = build_canonicalizer([triple])
    # Without the lattice there is NO functor overlap at all.
    assert cosine(functor_vector(triple.query), functor_vector(triple.analog)) == 0.0
    # With ancestor-closure features the MAC stage sees the bridge.
    qv = functor_vector(triple.query, canon=canon, delta=2)
    av = functor_vector(triple.analog, canon=canon, delta=2)
    assert cosine(qv, av) > 0.0
    # Full retrieval ranks the structural analog over the surface distractor
    # and stays certified-equal to brute force.
    index = MacFacIndex(config=ssb_config(), canon=canon)
    index.build([triple.analog, triple.distractor])
    retrieved = index.retrieve(triple.query, k=2, shortlist=2)
    assert retrieved[0].case_id == triple.analog.case_id
    brute = index.brute_force(triple.query, k=2)
    assert [r.case_id for r in retrieved] == [r.case_id for r in brute]


@pytest.mark.gate_G5
def test_agent_policies_and_service(tmp_path):
    with pytest.raises(ValueError):
        reject_free_text_facts("the service failed")
    service = MemoryService(tmp_path)
    encoded = service.encode("ERROR svc timeout\nWARN svc retry", "logs")
    results = service.retrieve(case_id=encoded["case_id"], k=1)
    assert results and results[0]["case_id"] == encoded["case_id"]
    assert service.verify("(timeout svc db)")["status"] in {"pass", "hypothetical"}
    assert verify_inference("(cause (timeout svc db) (retry svc db))").status in {"pass", "hypothetical"}


@pytest.mark.gate_G5
def test_comparison_modes_keep_llm_downstream_of_retrieval():
    framework = ComparisonFramework(orchestrator=LocalOrchestrator())
    framework.load_lines(demo_corpus(), adapter_id="logs", max_items=6)
    assert len(framework.items) == 6
    for mode in ["sma", "bm25", "dense rag", "knowledge graph", "context only"]:
        result = framework.ask("timeout caused retry storm", mode, adapter_id="logs", k=2)
        assert result.mode == mode
        assert result.evidence
        assert "Local LLM" in result.answer or result.llm_status.get("loaded") is True
    # Legacy aliases keep working.
    assert framework.ask("timeout", "rag", adapter_id="logs", k=2).mode == "dense rag"
    sma_result = framework.ask("timeout caused retry storm", "sma", adapter_id="logs", k=2)
    assert "case=" in sma_result.evidence[0]["provenance"]


@pytest.mark.gate_G6
def test_sage_probabilities_and_pool():
    pool = SagePool("test", assimilation_threshold=0.0)
    c1 = make_case([stmt("timeout", "svc", "db"), stmt("retry", "svc", "db")])
    c2 = make_case([stmt("timeout", "svc", "db"), stmt("retry", "svc", "db")])
    assert pool.assimilate(c1) == "outlier"
    assignment = pool.assimilate(c2)
    assert assignment.startswith("test_gen_")
    stats = pool.stats()
    assert stats["n_generalizations"] == 1
    assert support_probability(2, 2) == pytest.approx(0.75)


@pytest.mark.gate_G0
def test_report_fixture_rows_are_generated():
    rows = run_fixture_eval(library_n=12, mac_prefilter_n=20)
    assert rows["retrieval_runs.csv"]
    assert rows["mapping_runs.csv"]
    assert rows["ssb_cases.csv"]
    assert rows["ssb_metrics.csv"]
    metrics_by_split = {row["split"]: row for row in rows["ssb_metrics.csv"]}
    assert metrics_by_split["forced_choice_fixture"]["r1"] == "1.0000"
    assert metrics_by_split["forced_choice_fixture"]["mrr"] == "1.0000"
    assert metrics_by_split["ssb_library_12_sma"]["r1"] == "1.0000"
    assert "ssb_library_20_mac_prefilter" in metrics_by_split
    assert metrics_by_split["ssb_library_12_bm25"]["r1"] == "0.0000"
    assert metrics_by_split["ssb_library_12_tfidf_dense"]["r1"] == "0.0000"
