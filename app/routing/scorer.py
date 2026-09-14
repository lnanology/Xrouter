"""Scoring formula (section 八):

    score = capability_score * reliability_score * availability_score * confidence_score

then adjusted by policy weights over speed / cost / quota risk / local
preference. A candidate with availability_score == 0 (unhealthy, circuit
open, disabled, or quota-critical) is excluded outright, never merely
down-weighted.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.contracts.model import ModelInfo
from app.contracts.policy import PolicyWeights
from app.contracts.provider import ProviderHealth, ProviderStatus
from app.quota.limiter import weight_multiplier
from app.quota.tracker import QuotaRisk
from app.reliability.circuit_breaker import CircuitState


@dataclass
class ScoreInput:
    model: ModelInfo
    provider_health: ProviderHealth
    circuit_state: CircuitState
    quota_risk: QuotaRisk
    requires_streaming: bool = False
    requires_tools: bool = False
    # Phase 2: observed-performance weight from PerformanceController.
    # 1.0 (neutral) until it has enough samples to have an opinion.
    performance_weight: float = 1.0


def _availability_score(health: ProviderHealth, circuit_state: CircuitState) -> float:
    if circuit_state == CircuitState.OPEN:
        return 0.0
    if health.status == ProviderStatus.DISABLED or health.status == ProviderStatus.OFFLINE:
        return 0.0
    if health.status == ProviderStatus.COOLDOWN:
        return 0.0
    if circuit_state == CircuitState.HALF_OPEN:
        return 0.5
    if health.status == ProviderStatus.DEGRADED:
        return 0.6
    return 1.0


def _capability_score(inp: ScoreInput) -> float:
    if inp.requires_streaming and not inp.model.supports_streaming:
        return 0.0
    if inp.requires_tools and not inp.model.supports_tools:
        return 0.0
    return 1.0


def score_candidate(inp: ScoreInput, weights: PolicyWeights, is_local: bool) -> float | None:
    capability = _capability_score(inp)
    if capability <= 0:
        return None

    availability = _availability_score(inp.provider_health, inp.circuit_state)
    if availability <= 0:
        return None

    quota_mult = weight_multiplier(inp.quota_risk)
    if quota_mult <= 0:
        return None

    reliability = inp.model.reliability_score * inp.provider_health.success_rate
    confidence = inp.model.quality_score

    base = capability * reliability * availability * confidence

    # Policy adjustments.
    speed_component = inp.model.speed_score * weights.speed
    cost_component = inp.model.cost_score * weights.cost
    reliability_component = reliability * weights.reliability
    quality_component = inp.model.quality_score * weights.quality
    quota_component = quota_mult * weights.quota_risk
    local_component = (1.2 if is_local else 1.0) * weights.local_preference

    adjustment = (speed_component + cost_component + reliability_component + quality_component + quota_component) / 5.0

    score = base * adjustment * local_component

    # Latency penalty: gently discourage slow providers without hard-excluding them.
    if inp.provider_health.latency_ms:
        score *= 1.0 / (1.0 + (inp.provider_health.latency_ms / 2000.0))

    # Phase 2: fold in observed performance trend (gated by min sample count
    # inside PerformanceController itself, so this is a no-op — multiplier
    # 1.0 — until there's enough real telemetry to trust).
    score *= inp.performance_weight

    return max(score, 0.0)
