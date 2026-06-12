"""Inference verifier."""

from __future__ import annotations

from dataclasses import dataclass

from sma.ir.sexpr import loads_statement
from sma.ir.signatures import SignatureRegistry


@dataclass(frozen=True)
class VerificationResult:
    status: str
    reasons: tuple[str, ...] = ()


def verify_inference(inference_sexpr: str, registry: SignatureRegistry | None = None) -> VerificationResult:
    registry = registry or SignatureRegistry.with_defaults()
    try:
        statement = loads_statement(inference_sexpr)
        registry.validate_statement(statement)
    except Exception as exc:
        return VerificationResult("type_fail", (str(exc),))
    if "AnalogySkolemFn_" in inference_sexpr:
        return VerificationResult("hypothetical", ("contains analogy skolems",))
    return VerificationResult("pass", ())

