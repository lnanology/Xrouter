"""Generic OpenAI-compatible adapter. Works with any provider exposing the
standard /chat/completions and /models endpoints under `base_url`
(Groq, OpenRouter, and any self-hosted OpenAI-compatible server). Provider
identity/branding differences are handled entirely through config, not code.
"""
from __future__ import annotations

import json as _json
import time
from collections.abc import AsyncIterator

import httpx

from app.contracts.model import ModelInfo
from app.contracts.provider import ProviderCapability, ProviderConfig, ProviderHealth, ProviderStatus
from app.contracts.request import ChatCompletionRequest
from app.contracts.response import ChatCompletionChoice, ChatCompletionChunk, ChatCompletionChunkChoice, ChatCompletionResponse, Usage
from app.core.errors import (
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderInvalidRequestError,
    ProviderRateLimitError,
    ProviderServerError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.providers.base import Provider
from app.utils.secrets import read_secret


class OpenAICompatibleAdapter(Provider):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._api_key = read_secret(config.api_key_env)
        self._configured = bool(config.base_url) and (bool(self._api_key) or config.extra.get("no_auth"))
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        self._client = httpx.AsyncClient(
            base_url=config.base_url or "https://invalid.local",
            headers=headers,
            timeout=config.timeout_seconds,
        )

    def capabilities(self) -> set[ProviderCapability]:
        return {ProviderCapability.CHAT, ProviderCapability.STREAMING, ProviderCapability.TOOLS, ProviderCapability.EMBEDDINGS}

    def _require_configured(self) -> None:
        if not self._configured:
            raise ProviderUnavailableError(
                f"Provider '{self.id}' is missing base_url or API key ({self.config.api_key_env}); disabled.",
                provider_id=self.id,
            )

    async def list_models(self) -> list[ModelInfo]:
        if not self._configured:
            return []
        try:
            resp = await self._client.get("/models", timeout=5.0)
            if resp.status_code != 200:
                return []
            data = resp.json()
        except Exception:
            return []

        models: list[ModelInfo] = []
        for m in data.get("data", []):
            mid = m.get("id")
            if not mid:
                continue
            models.append(
                ModelInfo(
                    id=mid,
                    provider_id=self.id,
                    name=mid,
                    capabilities=["chat", "streaming"],
                    context_length=int(m.get("context_length", 8192) or 8192),
                    supports_streaming=True,
                    supports_tools=True,
                    supports_vision=False,
                    quality_score=float(self.config.extra.get("quality_score", 0.65)),
                    speed_score=float(self.config.extra.get("speed_score", 0.75)),
                    reliability_score=float(self.config.extra.get("reliability_score", 0.8)),
                    cost_score=float(self.config.extra.get("cost_score", 0.6)),
                )
            )
        return models

    async def health(self) -> ProviderHealth:
        if not self._configured:
            return ProviderHealth(status=ProviderStatus.DISABLED, last_checked=time.time(), last_error="not configured")
        start = time.time()
        try:
            resp = await self._client.get("/models", timeout=5.0)
            latency = (time.time() - start) * 1000
            if resp.status_code == 200:
                return ProviderHealth(status=ProviderStatus.HEALTHY, latency_ms=latency, last_checked=time.time())
            if resp.status_code == 429:
                return ProviderHealth(status=ProviderStatus.DEGRADED, latency_ms=latency, last_checked=time.time(), last_error="429")
            return ProviderHealth(
                status=ProviderStatus.DEGRADED, latency_ms=latency, last_checked=time.time(), last_error=f"HTTP {resp.status_code}"
            )
        except Exception as e:
            return ProviderHealth(status=ProviderStatus.OFFLINE, last_checked=time.time(), last_error=str(e))

    def _raise_for_status(self, resp: httpx.Response) -> None:
        if resp.status_code < 400:
            return
        text = resp.text[:300]
        if resp.status_code == 401 or resp.status_code == 403:
            raise ProviderAuthError(f"Auth failed ({resp.status_code}): {text}", provider_id=self.id)
        if resp.status_code == 429:
            retry_after = resp.headers.get("retry-after")
            raise ProviderRateLimitError(
                f"Rate limited: {text}", provider_id=self.id, retry_after=float(retry_after) if retry_after else None
            )
        if resp.status_code == 400:
            raise ProviderInvalidRequestError(f"Bad request: {text}", provider_id=self.id)
        if resp.status_code >= 500:
            raise ProviderServerError(f"HTTP {resp.status_code}: {text}", provider_id=self.id)
        raise ProviderError(f"HTTP {resp.status_code}: {text}", provider_id=self.id)

    async def chat(self, model: str, request: ChatCompletionRequest) -> ChatCompletionResponse:
        self._require_configured()
        self._usage.requests_total += 1
        body = request.model_dump(exclude_none=True, exclude={"routing_policy", "x_cache"})
        body["model"] = model
        body["stream"] = False
        try:
            resp = await self._client.post("/chat/completions", json=body)
        except httpx.TimeoutException as e:
            self._usage.requests_failed += 1
            raise ProviderTimeoutError(str(e), provider_id=self.id) from e
        except httpx.ConnectError as e:
            self._usage.requests_failed += 1
            raise ProviderConnectionError(str(e), provider_id=self.id) from e

        try:
            self._raise_for_status(resp)
        except ProviderError:
            self._usage.requests_failed += 1
            raise

        data = resp.json()
        choices = [
            ChatCompletionChoice(
                index=c.get("index", 0),
                message=c.get("message", {}),
                finish_reason=c.get("finish_reason"),
            )
            for c in data.get("choices", [])
        ]
        usage_raw = data.get("usage", {}) or {}
        usage = Usage(
            prompt_tokens=usage_raw.get("prompt_tokens", 0),
            completion_tokens=usage_raw.get("completion_tokens", 0),
            total_tokens=usage_raw.get("total_tokens", 0),
        )
        self._usage.tokens_total += usage.total_tokens
        return ChatCompletionResponse(model=f"{self.id}/{model}", choices=choices, usage=usage)

    async def stream_chat(self, model: str, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionChunk]:
        self._require_configured()
        self._usage.requests_total += 1
        body = request.model_dump(exclude_none=True, exclude={"routing_policy", "x_cache"})
        body["model"] = model
        body["stream"] = True

        try:
            async with self._client.stream("POST", "/chat/completions", json=body) as resp:
                if resp.status_code >= 400:
                    self._usage.requests_failed += 1
                    await resp.aread()
                    self._raise_for_status(resp)

                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:"):].strip()
                    if payload == "[DONE]":
                        return
                    if not payload:
                        continue
                    try:
                        data = _json.loads(payload)
                    except _json.JSONDecodeError:
                        continue
                    choices = [
                        ChatCompletionChunkChoice(
                            index=c.get("index", 0),
                            delta=c.get("delta", {}),
                            finish_reason=c.get("finish_reason"),
                        )
                        for c in data.get("choices", [])
                    ]
                    yield ChatCompletionChunk(id=data.get("id", "chatcmpl"), model=f"{self.id}/{model}", choices=choices)
        except httpx.TimeoutException as e:
            self._usage.requests_failed += 1
            raise ProviderTimeoutError(str(e), provider_id=self.id) from e
        except httpx.ConnectError as e:
            self._usage.requests_failed += 1
            raise ProviderConnectionError(str(e), provider_id=self.id) from e

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        self._require_configured()
        self._usage.requests_total += 1
        body = {"model": model, "input": texts}
        try:
            resp = await self._client.post("/embeddings", json=body)
        except httpx.TimeoutException as e:
            self._usage.requests_failed += 1
            raise ProviderTimeoutError(str(e), provider_id=self.id) from e
        except httpx.ConnectError as e:
            self._usage.requests_failed += 1
            raise ProviderConnectionError(str(e), provider_id=self.id) from e

        try:
            self._raise_for_status(resp)
        except ProviderError:
            self._usage.requests_failed += 1
            raise

        data = resp.json()
        rows = sorted(data.get("data", []), key=lambda r: r.get("index", 0))
        return [r.get("embedding", []) for r in rows]

    async def close(self) -> None:
        await self._client.aclose()
