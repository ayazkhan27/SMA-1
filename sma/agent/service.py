"""In-process SMA memory service used by CLI, API, and UI."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from sma.encoders import get_encoder
from sma.index.macfac import MacFacIndex
from sma.ir.schema import Case, make_case
from sma.ir.sexpr import canonical_case_text, loads_case
from sma.match.engine import match_cases
from sma.match.explain import correspondence_table, explain_text
from sma.match.infer import candidate_inferences
from sma.match.types import MatchConfig
from sma.match.verifier import verify_inference
from sma.sage.pools import SagePool
from sma.store import CaseStore

from .policies import reject_free_text_facts


class MemoryService:
    def __init__(self, store_path: str | Path = "data/processed/store"):
        self.store = CaseStore(store_path)
        self.config = MatchConfig(gamma=0.25, rho=0.5, delta=2)
        self.index = MacFacIndex(self.config)
        self.gmaps = {}
        self.pools: dict[str, SagePool] = {}
        for case in self.store.iter_cases():
            self.index.add(case)

    def encode(self, artifact: str, adapter_id: str, **kwargs) -> dict:
        result = get_encoder(adapter_id).encode(artifact, **kwargs)
        self.store.put(result.case)
        self.index.add(result.case)
        return {
            "case_id": result.case.case_id,
            "n_statements": len(result.case.statements),
            "warnings": list(result.warnings),
            "sexpr": canonical_case_text(result.case.statements),
        }

    def inline_case(self, sexpr: str) -> Case:
        return make_case(loads_case(sexpr), {"adapter": "inline"})

    def retrieve(self, case_id: str | None = None, inline_case: str | None = None, k: int = 10) -> list[dict]:
        query = self.store.get(case_id) if case_id else self.inline_case(inline_case or "")
        return [asdict(result) for result in self.index.retrieve(query, k=k)]

    def map(self, base_id: str, target_id: str, scorer: str = "ses") -> dict:
        config = MatchConfig(
            gamma=self.config.gamma,
            rho=self.config.rho,
            delta=self.config.delta,
            scorer=scorer,
        )
        gmap = match_cases(self.store.get(base_id), self.store.get(target_id), config=config)
        gmap_id = f"{base_id[:8]}_{target_id[:8]}_{scorer}"
        self.gmaps[gmap_id] = gmap
        return {
            "gmap_id": gmap_id,
            "correspondences": gmap.correspondences,
            "SES_n": gmap.normalized_score,
            "gap": gmap.optimality_gap,
            "kernels_used": len(gmap.kernels),
        }

    def project(self, gmap_id: str) -> list[dict]:
        return [asdict(inference) for inference in candidate_inferences(self.gmaps[gmap_id])]

    def verify(self, inference: str) -> dict:
        result = verify_inference(inference)
        return asdict(result)

    def store_annotations(self, case_id: str, outcome_annotations) -> dict:
        reject_free_text_facts(outcome_annotations)
        return {"status": "ok", "case_id": case_id}

    def generalize(self, pool_id: str, case_id: str) -> dict:
        pool = self.pools.setdefault(pool_id, SagePool(pool_id=pool_id, config=self.config))
        assignment = pool.assimilate(self.store.get(case_id))
        return {"pool_id": pool_id, "assignment": assignment}

    def pool_stats(self, pool_id: str) -> dict:
        pool = self.pools.setdefault(pool_id, SagePool(pool_id=pool_id, config=self.config))
        return pool.stats()

    def explain(self, gmap_id: str) -> dict:
        gmap = self.gmaps[gmap_id]
        return {"text": explain_text(gmap), "table": correspondence_table(gmap)}


default_service = MemoryService()

