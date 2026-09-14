"""Shared serialization helpers for provider/model info, used by both
api/health.py (public) and api/admin.py (authenticated)."""
from __future__ import annotations

from app.core.context import AppContext


def serialize_providers(ctx: AppContext) -> list[dict]:
    out = []
    for pid, pcfg in ctx.settings.providers.items():
        health = ctx.providers.health_of(pid)
        breaker = ctx.circuits.get(pid)
        quota_risk = ctx.quota.risk(pid)
        out.append(
            {
                "id": pid,
                "name": pcfg.name,
                "type": pcfg.type,
                "enabled": ctx.providers.is_enabled(pid),
                "configured": ctx.providers.get(pid) is not None,
                "status": health.status.value,
                "latency_ms": health.latency_ms,
                "circuit_state": breaker.state.value,
                "quota_risk": quota_risk.value,
                "model_count": len(ctx.models.for_provider(pid)),
            }
        )
    return out


def serialize_models(ctx: AppContext) -> list[dict]:
    return [
        {
            "id": m.public_id(),
            "provider": m.provider_id,
            "name": m.name,
            "status": m.status.value,
            "active": m.active,
            "context_length": m.context_length,
            "supports_streaming": m.supports_streaming,
            "supports_tools": m.supports_tools,
            "supports_vision": m.supports_vision,
            "quality_score": m.quality_score,
            "speed_score": m.speed_score,
            "reliability_score": m.reliability_score,
            "cost_score": m.cost_score,
        }
        for m in ctx.models.all()
    ]
