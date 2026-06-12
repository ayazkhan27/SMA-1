"""Convenience assimilation API."""

from __future__ import annotations

from sma.ir.schema import Case

from .pools import SagePool


def assimilate_stream(pool: SagePool, cases: list[Case]) -> list[str]:
    return [pool.assimilate(case) for case in cases]

