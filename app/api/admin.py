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
    available model and records TTFT/latency/success (section 三十五)."""
    ctx = request.app.state.context
    from app.contracts.request import ChatCompletionRequest, ChatMessage

    results = []
    test_request = ChatCompletionRequest(
        model="auto", messages=[ChatMessage(role="user", content="Reply with the single word: OK")], stream=False, max_tokens=8
    )
    for pid, provider in ctx.providers.all().items():
        if not ctx.providers.is_enabled(pid):
            continue
        models = ctx.models.for_provider(pid)
        if not models:
            continue
        model = models[0]
        start = time.time()
        success = False
        try:
            await provider.chat(model.name, test_request)
            success = True
        except Exception as e:
            logger.info("benchmark call failed for %s/%s: %s", pid, model.name, e)
        latency_ms = (time.time() - start) * 1000
        await ctx.metrics_repo.record_benchmark(pid, model.name, None, latency_ms, None, success)
        results.append({"provider": pid, "model": model.name, "latency_ms": round(latency_ms, 1), "success": success})
    return {"results": results}


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
