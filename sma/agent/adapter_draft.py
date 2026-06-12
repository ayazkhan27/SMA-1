"""LLM-drafted adapter rules: the model proposes RULES, never facts.

The LLM sees sample log lines and emits candidate keyword class rules as
strict JSON. The output is data for a deterministic encoder
(sma.encoders.draft_adapter.DraftAdapter); no model output ever enters a case
directly. Drafts are content-addressed (blake3) and stored with a
generated-by note so every case encoded under them is auditable; they remain
"LLM-proposed, unreviewed" until a human promotes them.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from sma.encoders.draft_adapter import (
    MAX_CLASSES,
    MAX_KEYWORDS,
    DraftRules,
    rules_hash,
    rules_to_json,
    validate_rules,
)
from sma.encoders.logs_drain import EVENT_CLASS_RULES, event_classes

# Keywords already claimed by the frozen ontology. A drafted keyword that
# equals one of these, or that contains one as a substring, can only fire on
# lines the frozen rules already cover - redundant rules double-count matches
# and dilute surprisal statistics (measured: the EOF rare-family regression).
FROZEN_KEYWORDS = tuple(sorted({kw for _, kws in EVENT_CLASS_RULES for kw in kws}))


def _redundant_keyword(keyword: str) -> bool:
    return any(frozen == keyword or frozen in keyword for frozen in FROZEN_KEYWORDS)

from .llm import DeepSeekOrchestrator, LocalOrchestrator, default_deepseek, default_orchestrator

MAX_SAMPLE_LINES = 30

DRAFT_DIR = Path("data/draft_adapters")

DRAFT_SYSTEM_PROMPT = (
    "You draft deterministic log-classification rules for SMA-1, a structure-mapping "
    "memory system. You propose RULES (keyword -> event class), never facts. Reply with "
    "STRICT JSON only - no prose, no markdown fences. Schema: "
    '{"classes": [{"name": "somethingEvent", "keywords": ["kw1", "kw2"]}], '
    '"maskings": ["regex", ...]}. Constraints: at most '
    f"{MAX_CLASSES} classes; at most {MAX_KEYWORDS} keywords per class; class names are "
    "alphanumeric ending in 'Event'; keywords are lowercase substrings that appear in the "
    "sample lines; maskings are regexes for variable tokens (ids, timestamps, counters). "
    "Do NOT reuse these frozen class names: timeoutEvent, retryEvent, ioEvent, memoryEvent, "
    "kernelEvent, networkEvent, authEvent, storageEvent, lifecycleEvent, failureEvent."
)


def _resolve_orchestrator(llm) -> LocalOrchestrator | DeepSeekOrchestrator:
    if isinstance(llm, str):
        if llm == "local":
            return default_orchestrator
        if llm == "deepseek":
            return default_deepseek
        raise ValueError(f"unknown llm backend: {llm!r}; expected 'local' or 'deepseek'")
    return llm


def build_draft_prompt(sample_texts: list[str], residual_only: bool = True) -> list[dict]:
    """Build the drafting prompt from sample lines.

    residual_only (default): only lines that fire NO frozen ontology class are
    shown to the LLM, so it is structurally impossible for the draft to
    re-cover vocabulary the frozen rules already handle. Returns messages with
    an empty user sample if everything is covered (caller should not draft).
    """
    lines: list[str] = []
    for text in sample_texts:
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if residual_only and event_classes(stripped.lower()):
                continue
            lines.append(stripped)
            if len(lines) >= MAX_SAMPLE_LINES:
                break
        if len(lines) >= MAX_SAMPLE_LINES:
            break
    sample = "\n".join(lines)
    user = (
        "Sample log lines that fired NO rule of the frozen ontology (the covered lines "
        "are deliberately excluded - do not re-cover known vocabulary):\n\n"
        f"{sample}\n\n"
        "Propose extra deterministic class rules (JSON only, schema above) so these lines "
        "fire shared cross-system event classes. Prefer FEW, COARSE, reusable categories."
    )
    return [
        {"role": "system", "content": DRAFT_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_draft_response(raw: str) -> tuple[DraftRules, str]:
    """Defensive parse of the LLM reply into DraftRules.

    Returns (rules, note). On any parse failure returns empty rules plus the
    error text. Invalid or colliding classes are dropped (recorded in the
    note) rather than failing the whole draft.
    """
    text = raw.strip()
    # Strip markdown fences and any prose around the first JSON object.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return DraftRules(), f"parse failure: no JSON object in reply: {raw[:200]!r}"
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        return DraftRules(), f"parse failure: {exc}"
    if not isinstance(payload, dict):
        return DraftRules(), f"parse failure: expected object, got {type(payload).__name__}"

    classes: list[tuple[str, tuple[str, ...]]] = []
    dropped: list[str] = []
    for row in (payload.get("classes") or [])[:MAX_CLASSES]:
        if not isinstance(row, dict):
            dropped.append(repr(row))
            continue
        name = row.get("name")
        keywords = tuple(
            k.strip().lower()
            for k in (row.get("keywords") or [])[:MAX_KEYWORDS]
            if isinstance(k, str) and k.strip()
        )
        stripped_redundant = tuple(k for k in keywords if _redundant_keyword(k))
        keywords = tuple(k for k in keywords if not _redundant_keyword(k))
        if stripped_redundant:
            dropped.append(f"{name}: redundant keywords {', '.join(stripped_redundant)}")
        if not keywords:
            dropped.append(f"{name} (fully covered by frozen ontology)")
            continue
        candidate = DraftRules(classes=((str(name), keywords),))
        if validate_rules(candidate):
            dropped.append(str(name))
            continue
        if any(existing == name for existing, _ in classes):
            dropped.append(f"{name} (duplicate)")
            continue
        classes.append((str(name), keywords))
    maskings = tuple(
        m for m in (payload.get("maskings") or []) if isinstance(m, str) and m.strip()
    )
    rules = DraftRules(classes=tuple(classes), maskings=maskings)
    errors = validate_rules(rules)
    if errors:
        # Bad masking regexes etc.: drop maskings and retry once without them.
        rules = DraftRules(classes=tuple(classes))
        errors = validate_rules(rules)
        if errors:
            return DraftRules(), "parse failure: " + "; ".join(errors)
    note = f"parsed {len(rules.classes)} class(es), {len(rules.maskings)} masking(s)"
    if dropped:
        note += f"; dropped invalid: {', '.join(dropped)}"
    return rules, note


def store_draft(rules: DraftRules, generated_by: str, note: str) -> tuple[str, Path]:
    """Persist the draft artifact under its blake3 content address."""
    digest = rules_hash(rules)
    DRAFT_DIR.mkdir(parents=True, exist_ok=True)
    path = DRAFT_DIR / f"{digest[:16]}.json"
    artifact = {
        "blake3": digest,
        "generated_by": generated_by,
        "status": "LLM-proposed, unreviewed",
        "created": datetime.now(timezone.utc).isoformat(),
        "note": note,
        "rules": json.loads(rules_to_json(rules)),
    }
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return digest, path


def draft_rules(sample_texts: list[str], llm="deepseek") -> tuple[DraftRules, str]:
    """Ask an LLM backend to draft extra class rules from sample lines.

    Returns (rules, note). On any failure the rules are empty and the note
    carries the error. On success the artifact is stored content-addressed
    under data/draft_adapters/ with a generated-by note.
    """
    orchestrator = _resolve_orchestrator(llm)
    messages = build_draft_prompt(sample_texts)
    if not messages[-1]["content"].split("\n\n")[1].strip():
        return DraftRules(), (
            "all sample lines already fire frozen ontology rules; nothing to draft "
            "(coverage is not the problem for this corpus)"
        )
    try:
        raw = orchestrator.complete(messages, max_tokens=600, temperature=0.0)
    except Exception as exc:
        return DraftRules(), f"draft failed ({orchestrator.name}): {type(exc).__name__}: {exc}"
    rules, note = parse_draft_response(raw)
    if not rules.classes:
        return rules, note
    generated_by = f"{orchestrator.name}:{orchestrator.status.get('model', '?')} via sma.agent.adapter_draft"
    digest, path = store_draft(rules, generated_by, note)
    return rules, f"{note}; blake3={digest}; stored={path}; generated-by {generated_by}"


__all__ = [
    "DRAFT_SYSTEM_PROMPT",
    "MAX_SAMPLE_LINES",
    "build_draft_prompt",
    "draft_rules",
    "parse_draft_response",
    "store_draft",
]
