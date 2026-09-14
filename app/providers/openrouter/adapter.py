"""OpenRouter adapter. OpenAI-compatible aggregator that fronts many models
(including free-tier ones); thin wrapper around the generic adapter."""
from __future__ import annotations

from app.contracts.provider import ProviderConfig
from app.providers.openai_compatible.adapter import OpenAICompatibleAdapter

_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterAdapter(OpenAICompatibleAdapter):
    def __init__(self, config: ProviderConfig):
        if not config.base_url:
            config.base_url = _DEFAULT_BASE_URL
        config.extra.setdefault("speed_score", 0.6)
        config.extra.setdefault("cost_score", 0.7)
        config.extra.setdefault("quality_score", 0.7)
        config.extra.setdefault("reliability_score", 0.7)
        super().__init__(config)
