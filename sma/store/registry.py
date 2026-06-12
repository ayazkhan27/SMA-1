"""Schema and adapter version registry."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Registry:
    adapters: dict[str, str] = field(default_factory=dict)
    score_versions: dict[str, dict] = field(default_factory=dict)

    def register_adapter(self, adapter_id: str, version: str) -> None:
        self.adapters[adapter_id] = version

    def register_score(self, score_id: str, config: dict) -> None:
        self.score_versions[score_id] = dict(config)

    @classmethod
    def defaults(cls) -> "Registry":
        registry = cls()
        for adapter in ("logs", "code", "traces", "structured", "agentobs", "prose_tier1"):
            registry.register_adapter(adapter, "0.1.0")
        registry.register_score("score-v1-draft", {"gamma": 0.25, "rho": 0.5, "delta": 2})
        return registry

