"""Encoder base classes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from sma.ir.schema import Case


@dataclass(frozen=True)
class EncodeResult:
    case: Case
    warnings: tuple[str, ...] = ()


class Encoder(Protocol):
    adapter_id: str
    version: str

    def encode(self, artifact: str, **kwargs) -> EncodeResult:
        ...

