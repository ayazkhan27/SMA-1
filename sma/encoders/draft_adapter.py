"""LLM-drafted adapter: rules as data, never facts.

The frozen ontology (logs_drain.EVENT_CLASS_RULES, tag ontology-v1) cannot be
edited, but the blueprint's section 4.1 discipline allows EXTRA deterministic
class rules supplied as data at the adapter boundary. An LLM may *propose*
those rules (see sma.agent.adapter_draft); it never encodes anything. Once
the rules exist as data, encoding is pure deterministic keyword matching:
identical input + identical rules => identical case bytes.

DraftAdapter composes the standard LogEncoder output with additional class
statements derived from the supplied DraftRules, mirroring the
``event_classes`` first-match-per-class semantics of the frozen rules.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import blake3

from sma.ir.schema import entity, make_case, stmt

from .base import EncodeResult
from .logs_drain import EVENT_CLASS_RULES, LogEncoder, infer_session

FROZEN_CLASS_NAMES = frozenset(name for name, _ in EVENT_CLASS_RULES)

_CLASS_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*Event$")

MAX_CLASSES = 8
MAX_KEYWORDS = 5


@dataclass(frozen=True)
class DraftRules:
    """Extra deterministic class rules, supplied as data.

    ``classes``: ordered (class_name, tuple-of-keywords) pairs, mirroring the
    shape of the frozen EVENT_CLASS_RULES. ``maskings``: optional regexes for
    variable tokens (ids, timestamps, counters), validated and carried in the
    content-addressed artifact for future template masking. They are NOT
    applied before keyword matching - class matching mirrors event_classes
    exactly (raw lowered line), and LLM-drafted masks routinely cover the very
    substrings the keywords need (e.g. ``code=[a-z0-9-]+`` vs ``grain-drift``).
    """

    classes: tuple[tuple[str, tuple[str, ...]], ...] = ()
    maskings: tuple[str, ...] = field(default_factory=tuple)


def validate_rules(rules: DraftRules) -> list[str]:
    """Return a list of validation errors (empty list means valid)."""
    errors: list[str] = []
    seen: set[str] = set()
    for name, keywords in rules.classes:
        if not isinstance(name, str) or not _CLASS_NAME_RE.match(name):
            errors.append(
                f"class name {name!r} must be alphanumeric with an 'Event' suffix"
            )
        if name in FROZEN_CLASS_NAMES:
            errors.append(
                f"class name {name!r} collides with the frozen ontology-v1 EVENT_CLASS_RULES"
            )
        if name in seen:
            errors.append(f"duplicate class name {name!r}")
        seen.add(name)
        if not keywords:
            errors.append(f"class {name!r} has no keywords")
        for keyword in keywords:
            if not isinstance(keyword, str) or not keyword or keyword != keyword.lower():
                errors.append(
                    f"class {name!r} keyword {keyword!r} must be a non-empty lowercase string"
                )
    for pattern in rules.maskings:
        try:
            re.compile(pattern)
        except re.error as exc:
            errors.append(f"masking regex {pattern!r} does not compile: {exc}")
    return errors


def rules_hash(rules: DraftRules) -> str:
    """Content address (blake3) of the canonical JSON form of the rules."""
    return blake3.blake3(rules_to_json(rules).encode("utf-8")).hexdigest()


def rules_to_json(rules: DraftRules) -> str:
    payload = {
        "classes": [
            {"name": name, "keywords": list(keywords)} for name, keywords in rules.classes
        ],
        "maskings": list(rules.maskings),
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def rules_from_json(text: str) -> DraftRules:
    payload = json.loads(text)
    classes = tuple(
        (row["name"], tuple(row["keywords"])) for row in payload.get("classes", [])
    )
    maskings = tuple(payload.get("maskings", []))
    return DraftRules(classes=classes, maskings=maskings)


class DraftAdapter:
    """LogEncoder output + extra deterministic class statements from DraftRules."""

    adapter_id = "logs+draft"
    version = "0.1.0"

    def __init__(self, rules: DraftRules):
        errors = validate_rules(rules)
        if errors:
            raise ValueError("invalid draft rules: " + "; ".join(errors))
        self.rules = rules
        self.draft_hash = rules_hash(rules)
        self._base = LogEncoder()

    def draft_classes(self, line: str) -> list[str]:
        """Mirror of logs_drain.event_classes, over the supplied rules.

        Matches the raw lowered line, exactly like the frozen rules; see the
        DraftRules docstring for why maskings are not applied here.
        """
        line_lower = line.lower()
        return [
            name
            for name, keywords in self.rules.classes
            if any(k in line_lower for k in keywords)
        ]

    def encode(self, artifact: str, **kwargs) -> EncodeResult:
        base = self._base.encode(artifact, **kwargs)
        session = kwargs.get("session_id") or infer_session(artifact)
        extra = []
        # Mirror the base encoder's event enumeration exactly: e{i} over
        # non-empty lines, so the extra class statements attach to the same
        # event entities the base statements use.
        for i, line in enumerate(line for line in artifact.splitlines() if line.strip()):
            for cls in self.draft_classes(line):
                extra.append(stmt(cls, entity(f"e{i}", "event"), entity(session, "session")))
        metadata = dict(base.case.metadata)
        metadata.update(
            {
                "adapter": "draft",
                "base_adapter": self._base.adapter_id,
                "draft_hash": self.draft_hash,
                "version": self.version,
            }
        )
        case = make_case(tuple(base.case.statements) + tuple(extra), metadata)
        return EncodeResult(case, base.warnings)


def check_determinism(adapter: DraftAdapter, text: str) -> bool:
    """Encode twice and assert identical canonical bytes (blueprint section 4 CI rule)."""
    from sma.ir.sexpr import canonical_case_text

    first = adapter.encode(text).case
    second = adapter.encode(text).case
    first_text = canonical_case_text(first.statements)
    second_text = canonical_case_text(second.statements)
    if first_text != second_text or first.case_id != second.case_id:
        raise AssertionError(
            f"draft adapter is non-deterministic for hash={adapter.draft_hash[:8]}"
        )
    return True


__all__ = [
    "DraftAdapter",
    "DraftRules",
    "MAX_CLASSES",
    "MAX_KEYWORDS",
    "check_determinism",
    "rules_from_json",
    "rules_hash",
    "rules_to_json",
    "validate_rules",
]
