"""The single interface every provider adapter must implement. Core code
(router, engine, fallback) only ever talks to this interface — never to a
concrete provider class."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from app.contracts.model import ModelInfo
from app.contracts.provider import ProviderCapability, ProviderConfig, ProviderHealth, ProviderUsage
from app.contracts.request import ChatCompletionRequest
from app.contracts.response import ChatCompletionChunk, ChatCompletionResponse
from app.core.errors import ProviderCapabilityUnsupportedError


class Provider(ABC):
    """Base class for all provider adapters.

    Implementations must be resilient to missing configuration: if an API key
    or local service is unavailable, `health()` should report OFFLINE rather
    than raising, and the adapter must not prevent the server from starting.
    """

    def __init__(self, config: ProviderConfig):
        self.config = config
        self._usage = ProviderUsage()

    @property
    def id(self) -> str:
        return self.config.id

    @property
    def name(self) -> str:
        return self.config.name

    @abstractmethod
    def capabilities(self) -> set[ProviderCapability]:
        """Static capability set this adapter type supports."""

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """Return the models this provider currently exposes. Must not raise;
        return [] on failure."""

    @abstractmethod
    async def health(self) -> ProviderHealth:
        """Cheap liveness/latency check. Must not raise."""

    @abstractmethod
    async def chat(self, model: str, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Non-streaming chat completion. Raises a subclass of
        app.core.errors.ProviderError on failure."""

    @abstractmethod
    def stream_chat(self, model: str, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionChunk]:
        """Streaming chat completion."""

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, one vector per input, same order.

        Concrete (not abstract) -- embeddings are an optional capability,
        like TOOLS/VISION, not something every adapter must implement.
        Default: unsupported. Only an adapter that actually declares
        ProviderCapability.EMBEDDINGS in capabilities() should override
        this -- and it must raise a subclass of
        app.core.errors.ProviderError on failure, never invent a fake
        vector."""
        raise ProviderCapabilityUnsupportedError(
            f"Provider '{self.id}' does not support embeddings", provider_id=self.id
        )

    def usage(self) -> ProviderUsage:
        return self._usage

    async def close(self) -> None:
        """Release any resources (HTTP clients, etc). Default no-op."""
        return None
