"""Routing decision contracts."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RoutingCandidate:
    provider_id: str
    model_id: str
    score: float
    reason: str = ""


@dataclass
class RoutingDecision:
    primary: RoutingCandidate
    fallback_chain: list[RoutingCandidate] = field(default_factory=list)
    policy: str = "balanced"
