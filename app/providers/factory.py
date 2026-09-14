"""The only place that maps a config `type` string to a concrete adapter
class. Core code never imports a concrete adapter directly — it goes through
`build_provider()`. Adding a new provider type means adding one line here,
not touching core/routing/reliability/etc."""
from __future__ import annotations

from app.contracts.provider import ProviderConfig
from app.providers.base import Provider
from app.providers.gemini.adapter import GeminiAdapter
from app.providers.groq.adapter import GroqAdapter
from app.providers.ollama.adapter import OllamaAdapter
from app.providers.openai_compatible.adapter import OpenAICompatibleAdapter
from app.providers.openrouter.adapter import OpenRouterAdapter

_ADAPTERS: dict[str, type[Provider]] = {
    "ollama": OllamaAdapter,
    "openai_compatible": OpenAICompatibleAdapter,
    "gemini": GeminiAdapter,
    "groq": GroqAdapter,
    "openrouter": OpenRouterAdapter,
}


def register_adapter(type_key: str, adapter_cls: type[Provider]) -> None:
    """Allows plugins to register new provider types without editing this
    file (Phase 2+ plugin loader will call this)."""
    _ADAPTERS[type_key] = adapter_cls


def build_provider(config: ProviderConfig) -> Provider:
    adapter_cls = _ADAPTERS.get(config.type)
    if adapter_cls is None:
        raise ValueError(f"Unknown provider type '{config.type}' for provider '{config.id}'")
    return adapter_cls(config)


def available_types() -> list[str]:
    return sorted(_ADAPTERS.keys())
