"""Shared matcher dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

from sma.ir.schema import Case, Entity, Node, Statement
from sma.ir.sexpr import dumps_node


def node_key(node: Node) -> str:
    prefix = "E" if isinstance(node, Entity) else "S"
    return f"{prefix}:{dumps_node(node)}"


@dataclass(frozen=True)
class MatchHypothesis:
    base: Node
    target: Node
    ascension: float = 1.0
    ancestor: str | None = None
    distance: int = 0

    # node_key serializes the whole expression tree, and these keys are read
    # O(kernels^2) times during merge — they must be computed once per instance.
    @cached_property
    def base_key(self) -> str:
        return node_key(self.base)

    @cached_property
    def target_key(self) -> str:
        return node_key(self.target)

    @cached_property
    def key(self) -> tuple[str, str]:
        return (self.base_key, self.target_key)


@dataclass
class Kernel:
    root: MatchHypothesis
    hypotheses: tuple[MatchHypothesis, ...]
    weight: float = 0.0

    @cached_property
    def bindings(self) -> dict[str, str]:
        return {mh.base_key: mh.target_key for mh in self.hypotheses}

    @cached_property
    def reverse_bindings(self) -> dict[str, str]:
        return {target: base for base, target in self.bindings.items()}


@dataclass
class GMap:
    base: Case
    target: Case
    hypotheses: tuple[MatchHypothesis, ...]
    kernels: tuple[Kernel, ...]
    score: float
    normalized_score: float
    scorer: str = "ses"
    optimality_gap: float | None = None

    @property
    def correspondences(self) -> list[dict[str, str | float | int | None]]:
        return [
            {
                "base": mh.base_key,
                "target": mh.target_key,
                "ascension": mh.ascension,
                "ancestor": mh.ancestor,
                "distance": mh.distance,
            }
            for mh in self.hypotheses
        ]


@dataclass(frozen=True)
class CandidateInference:
    inference_sexpr: str
    base_case_id: str
    target_case_id: str
    ses_n: float
    support: tuple[str, ...] = ()
    skolems: tuple[str, ...] = ()
    ascensions: tuple[str, ...] = ()
    status: str = "hypothetical"


@dataclass
class MatchConfig:
    gamma: float = 0.25
    rho: float = 0.95  # frozen at prereg-v1 (calibration grid; inert when delta=0)
    delta: int = 0
    scorer: str = "surprisal"  # "ses" | "mdl" | "surprisal" (score-v2, ADR-005)
    # Normalization of the structural score: "max" (blueprint 2.3),
    # "min" (10.2 tripwire), "sqrt" (geometric mean, cosine-style symmetric),
    # "target" (query-relative; ranking == raw-score ordering per query).
    # Frozen to "max" at prereg-v1 (calibration grid: beats target on family
    # and LOO-haystack validation). Registered caveat: out-of-corpus haystack
    # probes use hybrid fused as the production posture.
    normalization: str = "max"
    # Corpus surprisal per canonical functor (-log2 p), supplied by the index
    # for scorer="surprisal"; None means unit weights (identical to "ses").
    functor_costs: dict | None = None
    exact_kernel_limit: int = 60
    cpsat_time_ms: int = 20
    # Tripwire response from blueprint section 10.2: cap MH pairs per functor
    # group (U-ordered: identical statements first) so sessions with many
    # repeated event types cannot explode the kernel count quadratically.
    mh_group_cap: int = 128
    metadata: dict = field(default_factory=dict)

