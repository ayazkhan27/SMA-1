"""Standard concept-drift metrics adapted to the agent-memory setting."""
from __future__ import annotations


def update_recovery(correctness: list[int], change_idx: int) -> int | None:
    """Sessions after the change until the memory returns the NEW value and
    keeps it. None if it never recovers within the window."""
    for i in range(change_idx, len(correctness)):
        if correctness[i] == 1 and all(c == 1 for c in correctness[i:]):
            return i - change_idx + 1
    return None


def detection_delay(flags: list[bool], change_idx: int) -> int | None:
    """Sessions after the change until the detector first fires. None if never."""
    for i in range(change_idx, len(flags)):
        if flags[i]:
            return i - change_idx
    return None


def staleness_rate(correctness: list[int], change_idx: int) -> float:
    """Fraction of post-change probes that still returned the OLD value."""
    post = correctness[change_idx:]
    return 0.0 if not post else sum(1 for c in post if c == 0) / len(post)
