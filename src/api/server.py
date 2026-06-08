"""
server.py
=========
FastAPI service exposing semantic scene queries over HTTP.

Endpoints
---------
GET  /health          — Liveness check.
POST /query           — Text query → matching Gaussian positions + scores.
GET  /scene/info      — Number of loaded Gaussians and model metadata.

Usage (after building the semantic field)
-----------------------------------------
    uvicorn src.api.server:app --host 0.0.0.0 --port 8000

    curl -X POST http://localhost:8000/query \
         -H "Content-Type: application/json" \
         -d '{"text": "chair", "top_k": 500}'
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Video-to-3D Semantic Query API",
    description=(
        "Open-vocabulary 3D scene queries powered by LangSplat. "
        "Returns positions of 3D Gaussians that semantically match a text prompt."
    ),
    version="0.1.0",
)

# ── Lazy-load the semantic field at startup ───────────────────────────────────
# Paths are read from environment variables so the server is configurable
# without code changes.

_semantic_field = None


def _get_semantic_field():
    global _semantic_field
    if _semantic_field is None:
        from src.models.semantic_field import SemanticField

        gaussians_ply = os.environ.get("GAUSSIANS_PLY")
        ae_weights = os.environ.get("AE_WEIGHTS")

        if not gaussians_ply or not ae_weights:
            raise RuntimeError(
                "Set GAUSSIANS_PLY and AE_WEIGHTS environment variables before "
                "starting the server."
            )

        sf = SemanticField()
        sf.load(Path(gaussians_ply), Path(ae_weights))
        _semantic_field = sf
        logger.info("Semantic field loaded.")

    return _semantic_field


# ── Request / Response schemas ────────────────────────────────────────────────


class QueryRequest(BaseModel):
    text: str = Field(..., description="Open-vocabulary text query, e.g. 'chair'.")
    top_k: int = Field(1000, ge=1, le=50_000, description="Maximum Gaussians to return.")


class QueryResponse(BaseModel):
    query: str
    num_results: int
    positions: list[list[float]]   # list of [x, y, z]
    scores: list[float]


class SceneInfo(BaseModel):
    num_gaussians: int
    clip_model: str
    ae_latent_dim: int


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/scene/info", response_model=SceneInfo)
def scene_info():
    """Return metadata about the loaded semantic field."""
    try:
        sf = _get_semantic_field()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return SceneInfo(
        num_gaussians=len(sf._gaussians) if sf._gaussians is not None else 0,
        clip_model=sf.clip_model,
        ae_latent_dim=sf.ae_latent_dim,
    )


@app.post("/query", response_model=QueryResponse)
def semantic_query(req: QueryRequest):
    """Query the 3D scene with a natural-language text prompt.

    Returns the 3D positions (x, y, z) of Gaussians that best match the
    query, sorted by descending cosine similarity.
    """
    try:
        sf = _get_semantic_field()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        result = sf.query(req.text, top_k=req.top_k)
    except Exception as e:
        logger.exception("Query failed.")
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        query=result["query"],
        num_results=len(result["positions"]),
        positions=result["positions"],
        scores=result["scores"],
    )
