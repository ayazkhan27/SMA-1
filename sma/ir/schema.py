"""Typed predicate IR for SMA cases.

The implementation deliberately keeps the runtime representation small and
serializable. It is immutable enough for hashing, but remains plain Python so
fixtures and reports are easy to inspect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

import blake3


class SymbolKind(str, Enum):
    ENTITY = "entity"
    FUNCTION = "function"
    ATTRIBUTE = "attribute"
    RELATION = "relation"


@dataclass(frozen=True, slots=True)
class Signature:
    functor: str
    arity: int
    kind: SymbolKind = SymbolKind.RELATION
    arg_types: tuple[str, ...] = ()
    commutative: bool = False
    higher_order: bool = False

    def validate_arity(self, args: Sequence[Node]) -> None:
        if len(args) != self.arity:
            raise ValueError(f"{self.functor} expects {self.arity} args, got {len(args)}")


@dataclass(frozen=True, slots=True)
class Entity:
    name: str
    type: str = "entity"

    def nodes(self) -> tuple[Entity, ...]:
        return (self,)


@dataclass(frozen=True, slots=True)
class Statement:
    functor: str
    args: tuple["Node", ...] = ()
    ascension: float = 1.0

    def nodes(self) -> tuple["Node", ...]:
        out: list[Node] = [self]
        for arg in self.args:
            out.extend(arg.nodes())
        return tuple(out)

    def expressions(self) -> tuple["Statement", ...]:
        out: list[Statement] = [self]
        for arg in self.args:
            if isinstance(arg, Statement):
                out.extend(arg.expressions())
        return tuple(out)

    def entities(self) -> tuple[Entity, ...]:
        out: list[Entity] = []
        for arg in self.args:
            if isinstance(arg, Entity):
                out.append(arg)
            else:
                out.extend(arg.entities())
        return tuple(out)

    @property
    def arity(self) -> int:
        return len(self.args)


Node = Entity | Statement


@dataclass(frozen=True, slots=True)
class Case:
    case_id: str
    statements: tuple[Statement, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def expressions(self) -> tuple[Statement, ...]:
        seen: set[str] = set()
        ordered: list[Statement] = []
        from .sexpr import dumps_statement

        for stmt in self.statements:
            for expr in stmt.expressions():
                key = dumps_statement(expr)
                if key not in seen:
                    seen.add(key)
                    ordered.append(expr)
        return tuple(ordered)

    def entities(self) -> tuple[Entity, ...]:
        seen: set[tuple[str, str]] = set()
        ordered: list[Entity] = []
        for stmt in self.statements:
            for entity in stmt.entities():
                key = (entity.name, entity.type)
                if key not in seen:
                    seen.add(key)
                    ordered.append(entity)
        return tuple(ordered)

    def functor_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for expr in self.expressions():
            counts[expr.functor] = counts.get(expr.functor, 0) + 1
        return counts


def make_case(
    statements: Iterable[Statement], metadata: Mapping[str, Any] | None = None, case_id: str = ""
) -> Case:
    from .sexpr import canonical_case_text

    ordered = tuple(sorted(statements, key=lambda s: canonical_case_text([s])))
    text = canonical_case_text(ordered)
    return Case(case_id or content_id(text), ordered, dict(metadata or {}))


def content_id(text: str) -> str:
    # blake3 is a hard dependency: case ids are content addresses and must be
    # identical across machines, so no fallback hash is allowed here.
    return blake3.blake3(text.encode("utf-8")).hexdigest()


def walk_statement(stmt: Statement) -> Iterable[Node]:
    yield stmt
    for arg in stmt.args:
        if isinstance(arg, Statement):
            yield from walk_statement(arg)
        else:
            yield arg


def entity(name: str, type: str = "entity") -> Entity:
    return Entity(name=safe_symbol(name), type=safe_symbol(type))


def stmt(functor: str, *args: Node | str, ascension: float = 1.0) -> Statement:
    coerced = tuple(arg if isinstance(arg, (Entity, Statement)) else entity(str(arg)) for arg in args)
    return Statement(safe_symbol(functor), coerced, ascension=ascension)


def safe_symbol(value: str) -> str:
    value = str(value).strip()
    if not value:
        return "_"
    out = []
    for ch in value:
        if ch.isalnum() or ch in "_-:.@/":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)

