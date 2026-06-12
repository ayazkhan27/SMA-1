"""Code and bug encoder with Python AST fallback."""

from __future__ import annotations

import ast
import re

from sma.ir.schema import Statement, entity, make_case, stmt

from .base import EncodeResult


class CodeEncoder:
    adapter_id = "code"
    version = "0.1.0"

    def encode(self, artifact: str, **kwargs) -> EncodeResult:
        language = kwargs.get("language", "python")
        if language == "python":
            return self._encode_python(artifact)
        return self._encode_regex(artifact)

    def _encode_python(self, artifact: str) -> EncodeResult:
        statements: list[Statement] = []
        try:
            tree = ast.parse(artifact)
        except SyntaxError as exc:
            return EncodeResult(
                make_case([stmt("syntaxError", entity(str(exc.lineno or 0), "line"))], {"adapter": self.adapter_id, "tier": 0}),
                (str(exc),),
            )
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                statements.append(stmt("defines", entity(kind, "kind"), entity(node.name, kind)))
            elif isinstance(node, ast.Call):
                name = call_name(node.func)
                if name:
                    statements.append(stmt("calls", entity("module", "scope"), entity(name, "callable")))
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    statements.append(stmt("imports", entity("module", "scope"), entity(alias.name, "module")))
            elif isinstance(node, ast.Raise):
                statements.append(stmt("throws", entity("module", "scope"), entity(type(node).__name__, "exception")))
            elif isinstance(node, ast.ExceptHandler):
                exc = getattr(node.type, "id", "Exception") if node.type is not None else "Exception"
                statements.append(stmt("catches", entity("module", "scope"), entity(exc, "exception")))
        return EncodeResult(make_case(statements or [stmt("emptyCode", entity("module"))], {"adapter": self.adapter_id, "tier": 0}), ())

    def _encode_regex(self, artifact: str) -> EncodeResult:
        statements: list[Statement] = []
        for name in re.findall(r"\b(?:function|def|class)\s+([A-Za-z_]\w*)", artifact):
            statements.append(stmt("defines", entity("symbol", "kind"), entity(name, "symbol")))
        return EncodeResult(make_case(statements or [stmt("rawCode", entity("module"))], {"adapter": self.adapter_id, "tier": 0}), ())


def call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None

