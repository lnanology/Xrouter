"""Admin API — everything here requires a bearer token (section 三十三:
"Admin API 不可以公開裸奔"). If no XROUTER_ADMIN_TOKEN is set, a random token
is generated at startup and logged once."""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.api.providers import serialize_models, serialize_providers
from app.contracts.provider import ProviderStatus
from app.core.config import get_settings
from app.observability.logging import get_logger

router = APIRouter(prefix="/admin")
logger = get_logger("api.admin")
_bearer = HTTPBearer(auto_error=False)


def require_admin(request: Request, creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> None:
    ctx = request.app.state.context
    expected = ctx.settings.server.admin_token
    if not creds or creds.credentials != expected:
        raise HTTPException(status_code=401, detail="Missing or invalid admin token")


@router.get("/providers", dependencies=[Depends(require_admin)])
async def admin_providers(request: Request):
    ctx = request.app.state.context
    return {"providers": serialize_providers(ctx), "models": serialize_models(ctx)}


@router.get("/metrics", dependencies=[Depends(require_admin)])
async def admin_metrics(request: Request):
    ctx = request.app.state.context
    return {
        "metrics": ctx.metrics.snapshot(),
        "quota": ctx.quota.snapshot(),
        "circuit_breakers": {k: v.snapshot() for k, v in ctx.circuits.all().items()},
        "performance": ctx.performance.snapshot(),
    }


@router.post("/benchmark", dependencies=[Depends(require_admin)])
async def admin_benchmark(request: Request):
    """Sends one small prompt to each healthy, enabled provider's first
    available model and records TTFT/latency/success (section 三十五).
    The probe itself now lives in app/reliability/benchmark.py's
    BenchmarkScheduler (Phase 5's Automated Benchmark) so the same logic
    also runs on a schedule, not just from this manual trigger."""
    ctx = request.app.state.context
    return {"results": await ctx.benchmark_scheduler.run_once()}


@router.get("/benchmark/history", dependencies=[Depends(require_admin)])
async def admin_benchmark_history(request: Request, limit: int = 50):
    """Read-back for the benchmarks table -- MetricsRepository.
    recent_benchmarks() already existed but was never exposed anywhere."""
    ctx = request.app.state.context
    return {"benchmarks": await ctx.metrics_repo.recent_benchmarks(limit)}


@router.post("/reload", dependencies=[Depends(require_admin)])
async def admin_reload(request: Request):
    ctx = request.app.state.context
    ctx.settings = get_settings(reload=True)
    await ctx.models.refresh(ctx.providers)
    ctx.models.apply_overrides(ctx.settings.model_overrides)
    await ctx.model_repo.upsert_many(ctx.models.all())
    return {"reloaded": True, "models": len(ctx.models.all())}


@router.post("/providers/{provider_id}/enable", dependencies=[Depends(require_admin)])
async def enable_provider(provider_id: str, request: Request):
    ctx = request.app.state.context
    ctx.providers.set_enabled(provider_id, True)
    return {"provider_id": provider_id, "enabled": True}


@router.post("/providers/{provider_id}/disable", dependencies=[Depends(require_admin)])
async def disable_provider(provider_id: str, request: Request):
    ctx = request.app.state.context
    ctx.providers.set_enabled(provider_id, False)
    return {"provider_id": provider_id, "enabled": False}


@router.post("/providers/{provider_id}/cooldown", dependencies=[Depends(require_admin)])
async def cooldown_provider(provider_id: str, request: Request, seconds: float = 60.0):
    ctx = request.app.state.context
    breaker = ctx.circuits.get(provider_id)
    breaker.force_open(cooldown_seconds=seconds)

    from app.contracts.provider import ProviderHealth

    ctx.providers.set_health(provider_id, ProviderHealth(status=ProviderStatus.COOLDOWN, last_checked=time.time()))
    return {"provider_id": provider_id, "circuit_state": breaker.state.value, "cooldown_seconds": seconds}
