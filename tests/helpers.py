"""Test doubles shared across the suite. FakeProvider implements the same
Provider interface every real adapter implements, so router/fallback/
circuit-breaker tests never need real network access or Ollama installed."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from app.contracts.model import ModelInfo
from app.contracts.provider import ProviderCapability, ProviderConfig, ProviderHealth, ProviderStatus
from app.contracts.request import ChatCompletionRequest
from app.contracts.response import ChatCompletionChoice, ChatCompletionChunk, ChatCompletionChunkChoice, ChatCompletionResponse, Usage
from app.core.errors import ProviderError
from app.providers.base import Provider


class FakeProvider(Provider):
    """Scripted provider: `behavior` controls what chat()/stream_chat() do.

    behavior: "success" | "timeout" | "rate_limit" | "server_error" | "auth_error" | callable
    """

    def __init__(
        self, config: ProviderConfig, models: list[ModelInfo], behavior="success",
        fail_after_chunks: int | None = None, delay_seconds: float = 0.0,
        content: str = "ok", finish_reason: str = "stop",
    ):
        super().__init__(config)
        self._models = models
        self.behavior = behavior
        self.fail_after_chunks = fail_after_chunks
        self.delay_seconds = delay_seconds
        self.content = content
        self.finish_reason = finish_reason
        self.call_count = 0
        self.cancelled = False

    def capabilities(self) -> set[ProviderCapability]:
        return {ProviderCapability.CHAT, ProviderCapability.STREAMING}

    async def list_models(self) -> list[ModelInfo]:
        return self._models

    async def health(self) -> ProviderHealth:
        return ProviderHealth(status=ProviderStatus.HEALTHY, latency_ms=10.0, success_rate=1.0)

    def _maybe_raise(self):
        self.call_count += 1
        if callable(self.behavior):
            self.behavior(self.call_count)
            return
        if self.behavior == "timeout":
            from app.core.errors import ProviderTimeoutError

            raise ProviderTimeoutError("simulated timeout", provider_id=self.id)
        if self.behavior == "rate_limit":
            from app.core.errors import ProviderRateLimitError

            raise ProviderRateLimitError("simulated 429", provider_id=self.id, retry_after=0.01)
        if self.behavior == "server_error":
            from app.core.errors import ProviderServerError

            raise ProviderServerError("simulated 500", provider_id=self.id)
        if self.behavior == "auth_error":
            from app.core.errors import ProviderAuthError

            raise ProviderAuthError("simulated bad key", provider_id=self.id)

    async def chat(self, model: str, request: ChatCompletionRequest) -> ChatCompletionResponse:
        if self.delay_seconds:
            try:
                await asyncio.sleep(self.delay_seconds)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        self._maybe_raise()
        return ChatCompletionResponse(
            model=f"{self.id}/{model}",
            choices=[ChatCompletionChoice(index=0, message={"role": "assistant", "content": self.content}, finish_reason=self.finish_reason)],
            usage=Usage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
        )

    async def stream_chat(self, model: str, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionChunk]:
        self._maybe_raise()
        n_chunks = 3
        for i in range(n_chunks):
            if self.fail_after_chunks is not None and i >= self.fail_after_chunks:
                raise ProviderError("simulated mid-stream failure", provider_id=self.id)
            yield ChatCompletionChunk(
                id="chatcmpl-test",
                model=f"{self.id}/{model}",
                choices=[ChatCompletionChunkChoice(index=0, delta={"content": f"chunk{i}"}, finish_reason=None)],
            )


def make_model(provider_id: str, name: str = "test-model", **overrides) -> ModelInfo:
    defaults = dict(
        id=name, provider_id=provider_id, name=name, capabilities=["chat", "streaming"],
        supports_streaming=True, quality_score=0.7, speed_score=0.7, reliability_score=0.9, cost_score=0.8,
    )
    defaults.update(overrides)
    return ModelInfo(**defaults)
