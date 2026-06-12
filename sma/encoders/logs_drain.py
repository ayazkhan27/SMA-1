"""Deterministic Tier-0 log encoder.

This is a small Drain-like template masker for the MVP. When Drain3 is
installed, it can be substituted behind the same output contract.
"""

from __future__ import annotations

import re
from collections import Counter

from sma.ir.schema import Entity, Statement, entity, make_case, stmt

from .base import EncodeResult

IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
HEX_RE = re.compile(r"\b0x[0-9a-fA-F]+\b")
NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
BLOCK_RE = re.compile(r"\bblk[_-]?[A-Za-z0-9_-]+\b")

# Cross-system event ontology (blueprint section 4.3 mini-ontology, tripwire
# response to the measured ~100% cross-system lattice miss): template hashes
# are system-specific by construction, so every event also gets coarse
# deterministic class statements whose functors ARE shared across systems.
# First-match-per-class keyword rules over the lowercased line; a line may
# carry several classes. Order is fixed; rules are data.
EVENT_CLASS_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("timeoutEvent", ("timeout", "timed out")),
    ("retryEvent", ("retry", "retrying", "retransmit", "re-send", "resend")),
    ("ioEvent", ("eofexception", "ioexception", "io error", "input/output", "end of file")),
    ("memoryEvent", ("out of memory", "oom", "memory error", "ecc", "dimm", "cache error")),
    ("kernelEvent", ("kernel", "panic", "machine check", "mce", "interrupt")),
    ("networkEvent", ("connect", "socket", "network", "unreachable", "reset by peer",
                      "dhcp", "http", "link", "heartbeat", "packet")),
    ("authEvent", ("auth", "permission", "denied", "credential", "token", "login")),
    ("storageEvent", ("block", "replica", "disk", "volume", "snapshot", "image", "file system",
                      "filesystem", "storage")),
    ("lifecycleEvent", ("start", "stop", "restart", "boot", "shutdown", "terminat", "spawn",
                        "delet", "creat", "launch", "instance")),
    ("failureEvent", ("error", "fail", "exception", "fatal", "abort", "corrupt", "invalid")),
)


def event_classes(line_lower: str) -> list[str]:
    return [name for name, keywords in EVENT_CLASS_RULES if any(k in line_lower for k in keywords)]


class LogEncoder:
    adapter_id = "logs"
    version = "0.2.0"

    def encode(self, artifact: str, **kwargs) -> EncodeResult:
        session = kwargs.get("session_id") or infer_session(artifact)
        statements: list[Statement] = [stmt("logSession", entity(session, "session"))]
        events: list[Statement] = []
        counts: Counter[str] = Counter()
        for i, line in enumerate(line for line in artifact.splitlines() if line.strip()):
            template = mask_template(line)
            functor = template_functor(template)
            event_id = entity(f"e{i}", "event")
            event_stmt = stmt(functor, event_id, entity(session, "session"))
            events.append(event_stmt)
            counts[functor] += 1
            statements.append(event_stmt)
            component = infer_component(line)
            if component:
                statements.append(stmt("component", event_id, entity(component, "component")))
            line_lower = line.lower()
            if "timeout" in line_lower:
                statements.append(stmt("timeout", event_id, entity(session, "session")))
            if "retry" in line_lower:
                statements.append(stmt("retry", event_id, entity(session, "session")))
            if "error" in line_lower or "exception" in line_lower or "fail" in line_lower:
                statements.append(stmt("failureEvent", event_id, entity(session, "session")))
            for cls in event_classes(line_lower):
                statements.append(stmt(cls, event_id, entity(session, "session")))
        for left, right in zip(events, events[1:], strict=False):
            statements.append(stmt("before", left, right))
        for functor, count in counts.items():
            statements.append(stmt("count", entity(functor, "event_type"), entity(str(count), "integer")))
        statements.extend(derive_higher_order(events, statements))
        case = make_case(statements, {"adapter": self.adapter_id, "version": self.version, "tier": 0})
        return EncodeResult(case, ())


def mask_template(line: str) -> str:
    line = IP_RE.sub("<IP>", line)
    line = HEX_RE.sub("<HEX>", line)
    line = BLOCK_RE.sub("<BLOCK>", line)
    line = NUM_RE.sub("<NUM>", line)
    return " ".join(line.strip().split())


def template_functor(template: str) -> str:
    import hashlib

    digest = hashlib.blake2b(template.encode("utf-8"), digest_size=4).hexdigest()
    words = [w.lower() for w in re.findall(r"[A-Za-z]+", template)]
    alias = "evt"
    for key in ("timeout", "retry", "error", "exception", "restart", "fail", "block"):
        if key in words:
            alias = key
            break
    return f"{alias}_{digest}"


def infer_session(text: str) -> str:
    match = BLOCK_RE.search(text)
    if match:
        return match.group(0)
    return "session_0"


def infer_component(line: str) -> str | None:
    match = re.search(r"\b(?:INFO|WARN|ERROR|DEBUG)\s+([A-Za-z0-9_.-]+)", line)
    if match:
        return match.group(1)
    return None


def derive_higher_order(events: list[Statement], statements: list[Statement]) -> list[Statement]:
    out: list[Statement] = []
    timeouts = [s for s in statements if s.functor == "timeout"]
    retries = [s for s in statements if s.functor == "retry"]
    failures = [s for s in statements if s.functor == "failureEvent"]

    # rules/logs.yaml requires the antecedent event to precede the consequent
    # ("timeout before retry within session").
    def event_index(attribute: Statement) -> int:
        name = attribute.args[0].name if isinstance(attribute.args[0], Entity) else ""
        return int(name[1:]) if name.startswith("e") and name[1:].isdigit() else -1

    for timeout in timeouts:
        for retry in retries:
            if event_index(timeout) < event_index(retry):
                out.append(stmt("cause", timeout, retry))
    for failure in failures:
        for retry in retries:
            if event_index(failure) < event_index(retry):
                out.append(stmt("enables", failure, retry))
    return out

