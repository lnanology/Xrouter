"""Ollama adapter — the local / free / always-available fallback anchor.

Must never prevent server startup or crash the process if Ollama is not
installed or not running: health() reports OFFLINE and list_models()
returns [] in that case.
"""
from __future__ import annotations

import time
from collections.abc import AsyncIterator

import httpx

from app.contracts.model import ModelInfo
from app.contracts.provider import ProviderCapability, ProviderConfig, ProviderHealth, ProviderStatus
from app.contracts.request import ChatCompletionRequest
from app.contracts.response import ChatCompletionChoice, ChatCompletionChunk, ChatCompletionChunkChoice, ChatCompletionResponse, Usage
from app.core.errors import ProviderConnectionError, ProviderError, ProviderServerError, ProviderTimeoutError
from app.providers.base import Provider


def _to_ollama_messages(request: ChatCompletionRequest) -> list[dict]:
    out = []
    for m in request.messages:
        content = m.content if isinstance(m.content, str) else ("" if m.content is None else str(m.content))
        out.append({"role": m.role, "content": content})
    return out


class OllamaAdapter(Provider):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        base_url = config.base_url or "http://localhost:11434"
        self._client = httpx.AsyncClient(base_url=base_url, timeout=config.timeout_seconds)

    def capabilities(self) -> set[ProviderCapability]:
        return {ProviderCapability.CHAT, ProviderCapability.STREAMING, ProviderCapability.EMBEDDINGS}

    async def list_models(self) -> list[ModelInfo]:
        try:
            resp = await self._client.get("/api/tags", timeout=5.0)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return []

        models: list[ModelInfo] = []
        for m in data.get("models", []):
            name = m.get("name") or m.get("model")
            if not name:
                continue
            models.append(
                ModelInfo(
                    id=name,
                    provider_id=self.id,
                    name=name,
                    capabilities=["chat", "streaming"],
                    context_length=int((m.get("details") or {}).get("context_length", 8192) or 8192),
                    supports_streaming=True,
                    supports_tools=False,
                    supports_vision="vision" in name.lower() or "llava" in name.lower(),
                    quality_score=0.55,
                    speed_score=0.7,
                    reliability_score=0.9,  # local: no network flakiness
                    cost_score=1.0,  # free
                )
            )
        return models

    async def health(self) -> ProviderHealth:
        start = time.time()
        try:
            resp = await self._client.get("/api/tags", timeout=5.0)
            latency = (time.time() - start) * 1000
            if resp.status_code == 200:
                return ProviderHealth(status=ProviderStatus.HEALTHY, latency_ms=latency, last_checked=time.time())
            return ProviderHealth(
                status=ProviderStatus.DEGRADED,
                latency_ms=latency,
                last_checked=time.time(),
                last_error=f"HTTP {resp.status_code}",
            )
        except Exception as e:
            return ProviderHealth(
                status=ProviderStatus.OFFLINE,
                latency_ms=None,
                last_checked=time.time(),
                last_error=str(e),
            )

    async def chat(self, model: str, request: ChatCompletionRequest) -> ChatCompletionResponse:
        self._usage.requests_total += 1
        body = {
            "model": model,
            "messages": _to_ollama_messages(request),
            "stream": False,
        }
        if request.temperature is not None:
            body["options"] = {"temperature": request.temperature}
        try:
            resp = await self._client.post("/api/chat", json=body)
        except httpx.TimeoutException as e:
            self._usage.requests_failed += 1
            raise ProviderTimeoutError(str(e), provider_id=self.id) from e
        except httpx.ConnectError as e:
            self._usage.requests_failed += 1
            raise ProviderConnectionError(f"Cannot reach Ollama at {self._client.base_url}: {e}", provider_id=self.id) from e

        if resp.status_code >= 500:
            self._usage.requests_failed += 1
            raise ProviderServerError(f"Ollama HTTP {resp.status_code}", provider_id=self.id)
        if resp.status_code >= 400:
            self._usage.requests_failed += 1
            raise ProviderError(f"Ollama HTTP {resp.status_code}: {resp.text[:200]}", provider_id=self.id)

        data = resp.json()
        message = data.get("message", {"role": "assistant", "content": ""})
        prompt_tokens = int(data.get("prompt_eval_count", 0) or 0)
        completion_tokens = int(data.get("eval_count", 0) or 0)
        self._usage.tokens_total += prompt_tokens + completion_tokens

        return ChatCompletionResponse(
            model=f"{self.id}/{model}",
            choices=[ChatCompletionChoice(index=0, message=message, finish_reason="stop")],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    async def stream_chat(self, model: str, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionChunk]:
        self._usage.requests_total += 1
        body = {
            "model": model,
            "messages": _to_ollama_messages(request),
            "stream": True,
        }
        chunk_id = f"chatcmpl-{int(time.time()*1000)}"
        try:
            async with self._client.stream("POST", "/api/chat", json=body) as resp:
                if resp.status_code >= 400:
                    self._usage.requests_failed += 1
                    text = await resp.aread()
                    if resp.status_code >= 500:
                        raise ProviderServerError(f"Ollama HTTP {resp.status_code}", provider_id=self.id)
                    raise ProviderError(f"Ollama HTTP {resp.status_code}: {text[:200]}", provider_id=self.id)

                import json as _json

                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    data = _json.loads(line)
                    delta = data.get("message", {})
                    finish_reason = "stop" if data.get("done") else None
                    yield ChatCompletionChunk(
                        id=chunk_id,
                        model=f"{self.id}/{model}",
                        choices=[ChatCompletionChunkChoice(index=0, delta=delta, finish_reason=finish_reason)],
                    )
                    if data.get("done"):
                        completion_tokens = int(data.get("eval_count", 0) or 0)
                        prompt_tokens = int(data.get("prompt_eval_count", 0) or 0)
                        self._usage.tokens_total += prompt_tokens + completion_tokens
                        return
        except httpx.TimeoutException as e:
            self._usage.requests_failed += 1
            raise ProviderTimeoutError(str(e), provider_id=self.id) from e
        except httpx.ConnectError as e:
            self._usage.requests_failed += 1
            raise ProviderConnectionError(f"Cannot reach Ollama: {e}", provider_id=self.id) from e

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        self._usage.requests_total += 1
        body = {"model": model, "input": texts}
        try:
            resp = await self._client.post("/api/embed", json=body)
        except httpx.TimeoutException as e:
            self._usage.requests_failed += 1
            raise ProviderTimeoutError(str(e), provider_id=self.id) from e
        except httpx.ConnectError as e:
            self._usage.requests_failed += 1
            raise ProviderConnectionError(f"Cannot reach Ollama at {self._client.base_url}: {e}", provider_id=self.id) from e

        if resp.status_code >= 500:
            self._usage.requests_failed += 1
            raise ProviderServerError(f"Ollama HTTP {resp.status_code}", provider_id=self.id)
        if resp.status_code >= 400:
            self._usage.requests_failed += 1
            raise ProviderError(f"Ollama HTTP {resp.status_code}: {resp.text[:200]}", provider_id=self.id)

        data = resp.json()
        return data.get("embeddings", [])

    async def close(self) -> None:
        await self._client.aclose()
