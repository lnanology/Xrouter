"""GET /health, GET /health/providers — unauthenticated liveness endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Request

from app.api.providers import serialize_providers

router = APIRouter()


@router.get("/health")
async def health(request: Request):
    ctx = request.app.state.context
    providers = serialize_providers(ctx)
    any_healthy = any(p["status"] == "healthy" and p["enabled"] for p in providers)
    return {
        "status": "ok" if any_healthy else "degraded",
        "providers_enabled": sum(1 for p in providers if p["enabled"]),
        "providers_healthy": sum(1 for p in providers if p["status"] == "healthy"),
        "models_available": len(ctx.models.all()),
    }


@router.get("/health/providers")
async def health_providers(request: Request):
    ctx = request.app.state.context
    return {"providers": serialize_providers(ctx)}
