"""Provider-facing contracts. Core code depends only on these types, never on a
concrete provider implementation."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ProviderCapability(str, Enum):
    CHAT = "chat"
    STREAMING = "streaming"
    TOOLS = "tools"
    VISION = "vision"
    EMBEDDINGS = "embeddings"
    JSON_MODE = "json_mode"


class ProviderStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    COOLDOWN = "cooldown"
    DISABLED = "disabled"


@dataclass
class ProviderConfig:
    """Static, config-driven definition of a provider instance."""

    id: str
    name: str
    type: str  # adapter key, e.g. "ollama", "openai_compatible", "gemini"
    enabled: bool = True
    base_url: str | None = None
    api_key_env: str | None = None
    timeout_seconds: float = 30.0
    max_concurrency: int = 4
    priority: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderHealth:
    status: ProviderStatus = ProviderStatus.HEALTHY
    latency_ms: float | None = None
    success_rate: float = 1.0
    consecutive_failures: int = 0
    last_checked: float | None = None
    last_error: str | None = None


@dataclass
class ProviderUsage:
    requests_total: int = 0
    requests_failed: int = 0
    tokens_total: int = 0
    rate_limit_hits: int = 0
