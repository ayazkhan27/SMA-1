"""Canonical S-expression codec for SMA cases."""

from __future__ import annotations

from collections.abc import Iterable

from .schema import Entity, Node, Statement, entity


def dumps_node(node: Node) -> str:
    if isinstance(node, Entity):
        return node.name
    return dumps_statement(node)


def dumps_statement(statement: Statement) -> str:
    if not statement.args:
        return f"({statement.functor})"
    args = " ".join(dumps_node(arg) for arg in statement.args)
    return f"({statement.functor} {args})"


def canonical_case_text(statements: Iterable[Statement]) -> str:
    return "\n".join(sorted(dumps_statement(stmt) for stmt in statements))


def loads_statement(text: str) -> Statement:
    tokens = _tokenize(text)
    node, pos = _parse(tokens, 0)
    if pos != len(tokens):
        raise ValueError(f"trailing tokens after S-expression: {tokens[pos:]}")
    if not isinstance(node, Statement):
        raise ValueError("top-level S-expression must be a statement")
    return node


def loads_case(text: str) -> tuple[Statement, ...]:
    statements: list[Statement] = []
    tokens = _tokenize(text)
    pos = 0
    while pos < len(tokens):
        node, pos = _parse(tokens, pos)
        if not isinstance(node, Statement):
            raise ValueError("case entries must be statements")
        statements.append(node)
    return tuple(statements)


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    token = []
    for ch in text:
        if ch in "()":
            if token:
                tokens.append("".join(token))
                token = []
            tokens.append(ch)
        elif ch.isspace():
            if token:
                tokens.append("".join(token))
                token = []
        else:
            token.append(ch)
    if token:
        tokens.append("".join(token))
    return tokens


def _parse(tokens: list[str], pos: int) -> tuple[Node, int]:
    if pos >= len(tokens):
        raise ValueError("unexpected end of input")
    tok = tokens[pos]
    if tok != "(":
        return entity(tok), pos + 1
    if pos + 1 >= len(tokens):
        raise ValueError("missing functor")
    functor = tokens[pos + 1]
    args: list[Node] = []
    pos += 2
    while pos < len(tokens) and tokens[pos] != ")":
        arg, pos = _parse(tokens, pos)
        args.append(arg)
    if pos >= len(tokens) or tokens[pos] != ")":
        raise ValueError("unclosed statement")
    return Statement(functor, tuple(args)), pos + 1

