"""Model registry contracts."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ModelStatus(str, Enum):
    ACTIVE = "active"
    DEGRADED = "degraded"
    COOLDOWN = "cooldown"
    DISABLED = "disabled"
    DEPRECATED = "deprecated"


@dataclass
class ModelInfo:
    id: str
    provider_id: str
    name: str
    capabilities: list[str] = field(default_factory=list)
    context_length: int = 4096
    supports_streaming: bool = True
    supports_tools: bool = False
    supports_vision: bool = False

    # Static priors (config-seeded); adjusted over time from telemetry.
    quality_score: float = 0.5
    speed_score: float = 0.5
    reliability_score: float = 0.5
    cost_score: float = 0.5  # higher = cheaper

    active: bool = True
    status: ModelStatus = ModelStatus.ACTIVE

    def public_id(self) -> str:
        """The id clients see in /v1/models, e.g. 'ollama/llama3'."""
        return f"{self.provider_id}/{self.name}"
