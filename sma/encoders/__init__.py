from .agentobs import AgentObservationEncoder
from .base import EncodeResult
from .code_treesitter import CodeEncoder
from .logs_drain import LogEncoder
from .prose_tier1 import ProseTier1Encoder
from .structured import StructuredEncoder
from .healthcare import HealthcareEncoder
from .traces import TraceEncoder

ENCODERS = {
    "logs": LogEncoder,
    "code": CodeEncoder,
    "traces": TraceEncoder,
    "structured": StructuredEncoder,
    "healthcare": HealthcareEncoder,
    "agentobs": AgentObservationEncoder,
    "prose_tier1": ProseTier1Encoder,
}


def get_encoder(adapter_id: str):
    if adapter_id not in ENCODERS:
        raise KeyError(f"unknown adapter: {adapter_id}")
    return ENCODERS[adapter_id]()


__all__ = [
    "AgentObservationEncoder",
    "CodeEncoder",
    "ENCODERS",
    "EncodeResult",
    "LogEncoder",
    "ProseTier1Encoder",
    "StructuredEncoder",
    "TraceEncoder",
    "get_encoder",
]

