"""Stack trace and exception grammar encoder."""

from __future__ import annotations

import re

from sma.ir.schema import Statement, entity, make_case, stmt

from .base import EncodeResult

FRAME_RE = re.compile(r'File "?(?P<file>[^",\n]+)"?, line (?P<line>\d+), in (?P<func>[A-Za-z_][\w.]*)')
JAVA_FRAME_RE = re.compile(r"\s*at (?P<class>[\w.$]+)\.(?P<func>\w+)\((?P<file>[^:]+):(?P<line>\d+)\)")
CAUSE_RE = re.compile(r"(?:(?:Caused by|The above exception).*?:\s*)?(?P<exc>[A-Za-z_][\w.]*Error|[A-Za-z_][\w.]*Exception)")


class TraceEncoder:
    adapter_id = "traces"
    version = "0.1.0"

    def encode(self, artifact: str, **kwargs) -> EncodeResult:
        statements: list[Statement] = []
        frames: list[Statement] = []
        for i, line in enumerate(artifact.splitlines()):
            frame = FRAME_RE.search(line) or JAVA_FRAME_RE.search(line)
            if frame:
                file = frame.group("file")
                func = frame.group("func")
                line_no = frame.group("line")
                frame_stmt = stmt(
                    "frame",
                    entity(f"f{i}", "frame"),
                    entity(file, "file"),
                    entity(func, "function"),
                    entity(line_no, "line"),
                )
                frames.append(frame_stmt)
                statements.append(frame_stmt)
            cause = CAUSE_RE.search(line)
            if cause:
                statements.append(stmt("exception", entity(cause.group("exc"), "exception")))
        for left, right in zip(frames, frames[1:], strict=False):
            statements.append(stmt("calledFrom", left, right))
        case = make_case(statements or [stmt("emptyTrace", entity("trace_0"))], {"adapter": self.adapter_id, "tier": 0})
        return EncodeResult(case, ())

