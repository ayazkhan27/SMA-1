"""Signature registry and type validation."""

from __future__ import annotations

from dataclasses import dataclass, field

from .schema import Entity, Signature, Statement, SymbolKind


DEFAULT_SIGNATURES = (
    Signature("before", 2, higher_order=True),
    Signature("after", 2, higher_order=True),
    Signature("cause", 2, higher_order=True),
    Signature("implies", 2, higher_order=True),
    Signature("enables", 2, higher_order=True),
    Signature("prevents", 2, higher_order=True),
    Signature("during", 2, higher_order=True),
    Signature("count", 2),
    Signature("burst", 2),
    Signature("frame", 4),
    Signature("calledFrom", 2, higher_order=True),
    Signature("defines", 2),
    Signature("calls", 2),
    Signature("imports", 2),
    Signature("throws", 2),
    Signature("catches", 2),
    Signature("adds", 1),
    Signature("removes", 1),
    Signature("modifies", 1),
)


@dataclass
class SignatureRegistry:
    signatures: dict[str, Signature] = field(default_factory=dict)

    @classmethod
    def with_defaults(cls) -> "SignatureRegistry":
        registry = cls()
        for sig in DEFAULT_SIGNATURES:
            registry.register(sig)
        return registry

    def register(self, signature: Signature) -> None:
        self.signatures[signature.functor] = signature

    def get(self, functor: str, arity: int | None = None) -> Signature:
        if functor in self.signatures:
            sig = self.signatures[functor]
            if arity is not None and sig.arity != arity:
                raise ValueError(f"{functor} registered with arity {sig.arity}, got {arity}")
            return sig
        return Signature(functor=functor, arity=arity if arity is not None else -1)

    def validate_statement(self, statement: Statement) -> None:
        sig = self.get(statement.functor, statement.arity)
        if sig.arity >= 0:
            sig.validate_arity(statement.args)
        for arg in statement.args:
            if isinstance(arg, Statement):
                self.validate_statement(arg)
            elif not isinstance(arg, Entity):
                raise TypeError(f"unsupported argument node: {arg!r}")

    def validate_case(self, statements: tuple[Statement, ...]) -> None:
        for statement in statements:
            self.validate_statement(statement)


def infer_kind(functor: str, arity: int) -> SymbolKind:
    if arity == 0:
        return SymbolKind.ATTRIBUTE
    if arity == 1:
        return SymbolKind.ATTRIBUTE
    return SymbolKind.RELATION

