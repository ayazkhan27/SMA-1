"""Production-loop features: coverage tripwire, draft adapter, fused retrieval.

Covers blueprint 12-R3 (per-query lattice-miss/coverage indicator), the
LLM-drafted adapter (rules-as-data, deterministic encode), and the RRF-fused
hybrid mode with SME alignment receipts.
"""

from __future__ import annotations

import json

import pytest

from sma.agent.adapter_draft import build_draft_prompt, parse_draft_response
from sma.agent.comparison import MODE_ALIASES, MODES, ComparisonFramework, challenge_corpus
from sma.agent.llm import build_messages
from sma.encoders.coverage import COVERAGE_WARN_THRESHOLD, coverage_warning, rule_coverage
from sma.encoders.draft_adapter import (
    DraftAdapter,
    DraftRules,
    check_determinism,
    rules_from_json,
    rules_hash,
    rules_to_json,
    validate_rules,
)
from sma.encoders.logs_drain import EVENT_CLASS_RULES

WATCHER_LINES = (
    "2026-06-12T02:15:02.441Z WARN  zone=A7 unit=q-443 watcher.amber code=grain-drift-17 corrected=1 routed=1\n"
    "2026-06-12T02:16:01.004Z ERROR zone=A7 unit=q-443 watcher.red code=grain-drift-22 corrected=0 terminal=1\n"
    "2026-06-12T02:16:07.901Z WARN  zone=A7 unit=q-443 fence.state=closed"
)


# --- Feature 1: rule coverage ------------------------------------------------


def test_rule_coverage_fraction_counts():
    text = (
        "ERROR PaymentGateway connection timeout to db-shard-7\n"
        "\n"
        "frobnicator hums in zone A7\n"
        "WARN retrying transaction batch\n"
        "grain drift detected on q-443\n"
    )
    cov = rule_coverage(text)
    assert cov["total_lines"] == 4  # blank line excluded
    assert cov["covered_lines"] == 2  # timeout/connection line + retry line
    assert cov["fraction"] == pytest.approx(0.5)
    assert not cov["low"]
    assert coverage_warning(cov) is None


def test_rule_coverage_low_vocabulary_trips_warning():
    cov = rule_coverage("the moss sings\nthe fence hums\nquiet grain drift\n")
    assert cov["covered_lines"] == 0
    assert cov["fraction"] == 0.0
    assert cov["low"]
    warning = coverage_warning(cov)
    assert warning is not None
    assert "LOW (0%)" in warning
    assert "low-confidence" in warning


def test_rule_coverage_empty_text():
    cov = rule_coverage("")
    assert cov == {
        "fraction": 0.0,
        "covered_lines": 0,
        "total_lines": 0,
        "low": True,
        "percent": 0,
    }
    assert COVERAGE_WARN_THRESHOLD == 0.4


# --- Feature 2: draft rules validation + determinism ---------------------------


def good_rules() -> DraftRules:
    return DraftRules(
        classes=(
            ("watcherEvent", ("watcher.amber", "watcher.red")),
            ("grainDriftEvent", ("grain-drift",)),
        ),
        maskings=(r"\d{4}-\d{2}-\d{2}T[\d:.]+Z",),
    )


def test_validate_rules_accepts_good_rules():
    assert validate_rules(good_rules()) == []


def test_validate_rules_rejects_bad_shapes():
    frozen_name = EVENT_CLASS_RULES[0][0]
    bad = DraftRules(
        classes=(
            ("noSuffix", ("kw",)),  # missing Event suffix
            ("UPPERKeywordEvent", ("MixedCase",)),  # keyword not lowercase
            (frozen_name, ("kw",)),  # collides with frozen ontology
            ("emptyEvent", ()),  # no keywords
        ),
        maskings=("([unclosed",),  # bad regex
    )
    errors = validate_rules(bad)
    assert any("Event' suffix" in e for e in errors)
    assert any("lowercase" in e for e in errors)
    assert any("frozen" in e for e in errors)
    assert any("no keywords" in e for e in errors)
    assert any("does not compile" in e for e in errors)
    with pytest.raises(ValueError):
        DraftAdapter(bad)


def test_draft_adapter_deterministic_and_provenance():
    adapter = DraftAdapter(good_rules())
    assert check_determinism(adapter, WATCHER_LINES)
    case = adapter.encode(WATCHER_LINES).case
    # Provenance discipline: cases carry the draft stamp.
    assert case.metadata["adapter"] == "draft"
    assert case.metadata["draft_hash"] == adapter.draft_hash
    assert len(adapter.draft_hash) == 64
    # Extra class statements fired alongside the base encoder's statements.
    functors = {s.functor for s in case.statements}
    assert "watcherEvent" in functors
    assert "grainDriftEvent" in functors
    assert "logSession" in functors  # base LogEncoder output is preserved
    # Hash is a content address of the rules.
    assert adapter.draft_hash == rules_hash(good_rules())
    roundtrip = rules_from_json(rules_to_json(good_rules()))
    assert roundtrip == good_rules()


def test_maskings_do_not_erase_keyword_matches():
    # Regression from the live DeepSeek draft: a masking like code=[a-z0-9-]+
    # covers the very substring the grain-drift keyword needs. Class matching
    # must mirror event_classes (raw lowered line), with maskings carried as
    # data only.
    rules = DraftRules(
        classes=(("grainDriftEvent", ("grain-drift",)),),
        maskings=(r"code=[a-z0-9-]+",),
    )
    adapter = DraftAdapter(rules)
    line = "WARN zone=A7 unit=q-443 watcher.amber code=grain-drift-17 corrected=1"
    assert adapter.draft_classes(line) == ["grainDriftEvent"]


def test_parse_draft_response_defensive():
    raw = """```json
    {"classes": [
        {"name": "watcherEvent", "keywords": ["Watcher.Amber", "watcher.red"]},
        {"name": "timeoutEvent", "keywords": ["frozen-collision"]},
        {"name": "bad name!", "keywords": ["x"]}
    ], "maskings": ["q-\\\\d+"]}
    ```"""
    rules, note = parse_draft_response(raw)
    assert rules.classes == (("watcherEvent", ("watcher.amber", "watcher.red")),)
    assert rules.maskings == ("q-\\d+",)
    assert "dropped invalid" in note
    # Outright garbage -> empty rules + error, never an exception.
    rules, note = parse_draft_response("no json here at all")
    assert rules == DraftRules()
    assert "parse failure" in note


def test_build_draft_prompt_caps_sample_lines():
    texts = [f"line {i} grain" for i in range(100)]
    messages = build_draft_prompt(texts)
    assert messages[0]["role"] == "system"
    assert len([l for l in messages[1]["content"].splitlines() if l.startswith("line ")]) == 30


def test_framework_draft_adapter_apply_and_revert():
    framework = ComparisonFramework()
    framework.load_lines(challenge_corpus(), adapter_id="logs")
    base_case_ids = [item.case.case_id for item in framework.items]
    adapter = DraftAdapter(good_rules())
    count = framework.apply_draft_adapter(adapter)
    assert count == len(framework.items)
    assert framework.draft_note == (
        f"draft-adapter (LLM-proposed, unreviewed) hash={adapter.draft_hash[:8]}"
    )
    for item in framework.items:
        assert item.case.metadata["adapter"] == "draft"
        assert item.case.metadata["draft_hash"] == adapter.draft_hash
    evidence = framework.sma_evidence("ERROR gateway timeout then retry then failure", "logs", 3)
    rows = [row for row in evidence if not row.get("warning")]
    assert rows and all(framework.draft_note in row["mode_detail"] for row in rows)
    framework.revert_draft_adapter()
    assert framework.draft_note is None
    assert [item.case.case_id for item in framework.items] == base_case_ids


# --- Feature 1+3 through the framework ----------------------------------------


@pytest.fixture(scope="module")
def loaded_framework() -> ComparisonFramework:
    framework = ComparisonFramework()
    framework.load_lines(challenge_corpus(), adapter_id="logs")
    return framework


def test_sma_evidence_low_coverage_prepends_warning(loaded_framework):
    evidence = loaded_framework.sma_evidence("the moss sings of quiet grain drift", "logs", 3)
    assert evidence[0]["source_id"] == "coverage-tripwire"
    assert "LOW (0%)" in evidence[0]["warning"]
    assert evidence[0]["coverage"]["low"]
    # Verbalizer prompt carries the caveat but not the pseudo-row as evidence.
    user = build_messages("q", "sma", evidence)[-1]["content"]
    assert "structural coverage of this query is LOW (0%)" in user
    assert "coverage-tripwire" not in user


def test_sma_evidence_high_coverage_no_warning(loaded_framework):
    query = "ERROR gateway connection timeout\nWARN retrying batch\nERROR failed after retry"
    evidence = loaded_framework.sma_evidence(query, "logs", 3)
    assert all(not row.get("warning") for row in evidence)
    assert evidence and not evidence[0]["coverage"]["low"]


def test_hybrid_mode_registered():
    assert "hybrid (fused)" in MODES
    assert MODE_ALIASES["hybrid"] == "hybrid (fused)"


def test_hybrid_fused_evidence_shape(loaded_framework):
    query = (
        "ERROR StreamIngest connector timeout polling source kafka-9\n"
        "WARN StreamIngest connector retrying poll\n"
        "ERROR StreamIngest sink write failed after repeated retry"
    )
    mode, evidence = loaded_framework.evidence_for(query, "hybrid", k=4)
    assert mode == "hybrid (fused)"
    rows = [row for row in evidence if not row.get("warning")]
    assert 0 < len(rows) <= 4
    for row in rows:
        # Accountability rides on every fused candidate: SME receipts present.
        assert row["mode_detail"] == "RRF(bm25+dense+sma) candidates, SME alignment receipts"
        assert row["alignment"]
        assert "inferences" in row
        assert row["provenance"].startswith("rrf=")
        assert "ranks(bm25=" in row["provenance"]
        assert float(row["score"]) > 0
        assert "coverage" in row
    # Fused scores are descending.
    scores = [float(row["score"]) for row in rows]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_runs_through_ask_with_fallback_llm(loaded_framework):
    result = loaded_framework.ask("ERROR timeout then retry storm", "hybrid (fused)", k=3)
    assert result.mode == "hybrid (fused)"
    assert result.answer
    assert result.evidence
