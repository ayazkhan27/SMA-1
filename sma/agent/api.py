"""FastAPI tool surface for SMA."""

from __future__ import annotations

from typing import Any

from .service import default_service

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except ImportError:  # pragma: no cover - optional dependency fallback
    FastAPI = None  # type: ignore
    BaseModel = object  # type: ignore


if FastAPI is not None:
    app = FastAPI(title="SMA-1 Agentic Memory API")

    class EncodeRequest(BaseModel):
        artifact: str
        adapter_id: str
        kwargs: dict[str, Any] = {}

    class RetrieveRequest(BaseModel):
        case_id: str | None = None
        inline_case: str | None = None
        k: int = 10

    class MapRequest(BaseModel):
        base_id: str
        target_id: str
        scorer: str = "ses"

    class ProjectRequest(BaseModel):
        gmap_id: str

    class VerifyRequest(BaseModel):
        inference: str

    @app.post("/encode")
    def encode(req: EncodeRequest):
        return default_service.encode(req.artifact, req.adapter_id, **req.kwargs)

    @app.post("/retrieve")
    def retrieve(req: RetrieveRequest):
        return default_service.retrieve(req.case_id, req.inline_case, req.k)

    @app.post("/map")
    def map_cases(req: MapRequest):
        return default_service.map(req.base_id, req.target_id, req.scorer)

    @app.post("/project")
    def project(req: ProjectRequest):
        return default_service.project(req.gmap_id)

    @app.post("/verify")
    def verify(req: VerifyRequest):
        return default_service.verify(req.inference)

    @app.get("/pool/{pool_id}")
    def pool_stats(pool_id: str):
        return default_service.pool_stats(pool_id)

else:
    app = None

