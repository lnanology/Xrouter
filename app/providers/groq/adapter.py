"""Groq adapter. Groq's API is OpenAI-compatible, so this is a thin wrapper
around the generic adapter with Groq-appropriate defaults (fast inference,
generous free tier historically -> higher speed_score prior)."""
from __future__ import annotations

from app.contracts.provider import ProviderConfig
from app.providers.openai_compatible.adapter import OpenAICompatibleAdapter

_DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"


class GroqAdapter(OpenAICompatibleAdapter):
    def __init__(self, config: ProviderConfig):
        if not config.base_url:
            config.base_url = _DEFAULT_BASE_URL
        config.extra.setdefault("speed_score", 0.9)
        config.extra.setdefault("cost_score", 0.85)
        config.extra.setdefault("quality_score", 0.65)
        config.extra.setdefault("reliability_score", 0.75)
        super().__init__(config)
