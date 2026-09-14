"""Provider and model registries. Built once at startup from config; can be
rebuilt via POST /admin/reload. Never lets one bad provider config prevent
the others (or the server) from starting."""
from __future__ import annotations

import asyncio

from app.contracts.model import ModelInfo, ModelStatus
from app.contracts.provider import ProviderConfig, ProviderHealth, ProviderStatus
from app.core.config import Settings
from app.observability.logging import get_logger
from app.providers.base import Provider
from app.providers.factory import build_provider

logger = get_logger("registry")


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, Provider] = {}
        self._health: dict[str, ProviderHealth] = {}
        self._configs: dict[str, ProviderConfig] = {}
        self._enabled_override: dict[str, bool] = {}

    @classmethod
    def build(cls, settings: Settings) -> "ProviderRegistry":
        reg = cls()
        for pid, pcfg in settings.providers.items():
            reg._configs[pid] = pcfg
            if not pcfg.enabled:
                reg._health[pid] = ProviderHealth(status=ProviderStatus.DISABLED)
                continue
            try:
                provider = build_provider(pcfg)
            except Exception as e:  # a bad single provider must never crash startup
                logger.warning("failed to initialize provider '%s': %s", pid, e)
                reg._health[pid] = ProviderHealth(status=ProviderStatus.OFFLINE, last_error=str(e))
                continue
            reg._providers[pid] = provider
            reg._health[pid] = ProviderHealth(status=ProviderStatus.HEALTHY)
        return reg

    def register(self, provider: Provider, config: ProviderConfig | None = None, enabled: bool = True) -> None:
        """Manually register an already-constructed provider (used by tests
        and, in future, by the plugin loader for dynamically-added
        providers)."""
        pid = provider.id
        self._providers[pid] = provider
        self._configs[pid] = config or provider.config
        self._enabled_override[pid] = enabled
        self._health[pid] = ProviderHealth(status=ProviderStatus.HEALTHY)

    def get(self, provider_id: str) -> Provider | None:
        return self._providers.get(provider_id)

    def all(self) -> dict[str, Provider]:
        return dict(self._providers)

    def is_enabled(self, provider_id: str) -> bool:
        if provider_id in self._enabled_override:
            return self._enabled_override[provider_id]
        cfg = self._configs.get(provider_id)
        return bool(cfg and cfg.enabled)

    def set_enabled(self, provider_id: str, enabled: bool) -> None:
        self._enabled_override[provider_id] = enabled

    def health_of(self, provider_id: str) -> ProviderHealth:
        return self._health.get(provider_id, ProviderHealth(status=ProviderStatus.OFFLINE))

    def set_health(self, provider_id: str, health: ProviderHealth) -> None:
        self._health[provider_id] = health

    def all_health(self) -> dict[str, ProviderHealth]:
        return dict(self._health)

    async def close_all(self) -> None:
        await asyncio.gather(*(p.close() for p in self._providers.values()), return_exceptions=True)


class ModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, ModelInfo] = {}

    async def refresh(self, provider_registry: ProviderRegistry) -> None:
        results = await asyncio.gather(
            *(self._refresh_one(pid, p) for pid, p in provider_registry.all().items() if provider_registry.is_enabled(pid)),
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, Exception):
                logger.warning("model refresh error: %s", r)

    async def _refresh_one(self, provider_id: str, provider: Provider) -> None:
        try:
            models = await provider.list_models()
        except Exception as e:
            logger.warning("list_models failed for '%s': %s", provider_id, e)
            return
        for m in models:
            self._models[m.public_id()] = m

    def apply_overrides(self, overrides: dict[str, dict]) -> None:
        """Applies score/flag overrides from config/models.yaml, keyed by
        public_id ('ollama/llama3'). Lets operators tune routing without
        touching code."""
        for public_id, fields in overrides.items():
            model = self._models.get(public_id)
            if model is None:
                continue
            for key in ("quality_score", "speed_score", "reliability_score", "cost_score", "active"):
                if key in fields:
                    setattr(model, key, fields[key])

    def get(self, public_id: str) -> ModelInfo | None:
        return self._models.get(public_id)

    def all(self) -> list[ModelInfo]:
        return list(self._models.values())

    def for_provider(self, provider_id: str) -> list[ModelInfo]:
        return [m for m in self._models.values() if m.provider_id == provider_id]

    def set_status(self, public_id: str, status: ModelStatus) -> None:
        m = self._models.get(public_id)
        if m:
            m.status = status
            m.active = status == ModelStatus.ACTIVE
