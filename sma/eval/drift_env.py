"""Seeded synthetic ops drift environment."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class DriftStep:
    t: int
    event: str
    ground_truth: dict[str, str]


def generate_drift_world(steps: int = 20, seed: int = 7) -> list[DriftStep]:
    rng = random.Random(seed)
    services = ["api", "db", "queue", "worker"]
    states = {svc: "ok" for svc in services}
    out: list[DriftStep] = []
    for t in range(steps):
        svc = rng.choice(services)
        states[svc] = rng.choice(["ok", "timeout", "saturated", "restarting"])
        out.append(DriftStep(t=t, event=f"{svc} -> {states[svc]}", ground_truth=dict(states)))
    return out

