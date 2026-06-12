"""Agent observation encoder."""

from __future__ import annotations

from sma.ir.schema import entity, make_case, stmt

from .base import EncodeResult


class AgentObservationEncoder:
    adapter_id = "agentobs"
    version = "0.1.0"

    def encode(self, artifact: str, **kwargs) -> EncodeResult:
        command = kwargs.get("command", "tool")
        exit_code = str(kwargs.get("exit_code", 0))
        obs = entity("obs_0", "observation")
        statements = [
            stmt("toolOutput", obs, entity(command, "command")),
            stmt("exitCode", obs, entity(exit_code, "integer")),
        ]
        if "error" in artifact.lower() or exit_code != "0":
            statements.append(stmt("failureEvent", obs, entity(command, "command")))
        if artifact.strip():
            statements.append(stmt("outputDigest", obs, entity(str(abs(hash(artifact))), "digest")))
        return EncodeResult(make_case(statements, {"adapter": self.adapter_id, "tier": 0}), ())

