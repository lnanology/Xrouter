"""Routing policy contracts."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RoutingPolicyName(str, Enum):
    FASTEST = "fastest"
    CHEAPEST = "cheapest"
    RELIABLE = "reliable"
    QUOTA_AWARE = "quota_aware"
    QUALITY = "quality"
    BALANCED = "balanced"


@dataclass
class PolicyWeights:
    """Weights applied on top of the base capability/reliability/availability
    score. A policy is just a different set of weights + tie-break rule."""

    quality: float = 1.0
    speed: float = 1.0
    reliability: float = 1.0
    cost: float = 1.0
    quota_risk: float = 1.0
    local_preference: float = 1.0
