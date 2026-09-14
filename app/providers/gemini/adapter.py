"""Google Gemini adapter (generativelanguage.googleapis.com REST API).
Gracefully disables itself if GEMINI_API_KEY (or configured env var) is not
set — never crashes the server."""
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
    ProviderRateLimitError,
    ProviderServerError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.providers.base import Provider
from app.utils.secrets import read_secret

_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


def _to_gemini_contents(request: ChatCompletionRequest) -> tuple[list[dict], str | None]:
    system_instruction = None
    contents = []
    for m in request.messages:
        text = m.content if isinstance(m.content, str) else ("" if m.content is None else str(m.content))
        if m.role == "system":
            system_instruction = text
            continue
        role = "model" if m.role == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": text}]})
    return contents, system_instruction


class GeminiAdapter(Provider):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._api_key = read_secret(config.api_key_env)
        self._configured = bool(self._api_key)
        self._client = httpx.AsyncClient(base_url=config.base_url or _DEFAULT_BASE_URL, timeout=config.timeout_seconds)

    def capabilities(self) -> set[ProviderCapability]:
        return {ProviderCapability.CHAT, ProviderCapability.STREAMING, ProviderCapability.VISION}

    def _require_configured(self) -> None:
        if not self._configured:
            raise ProviderUnavailableError(
                f"Provider '{self.id}' missing API key ({self.config.api_key_env}); disabled.", provider_id=self.id
            )

    async def list_models(self) -> list[ModelInfo]:
        if not self._configured:
            return []
        try:
            resp = await self._client.get("/models", params={"key": self._api_key}, timeout=5.0)
            if resp.status_code != 200:
                return []
            data = resp.json()
        except Exception:
            return []
        models: list[ModelInfo] = []
        for m in data.get("models", []):
            full_name = m.get("name", "")
            mid = full_name.split("/")[-1] if full_name else None
            if not mid or "generateContent" not in (m.get("supportedGenerationMethods") or []):
                continue
            models.append(
                ModelInfo(
                    id=mid,
                    provider_id=self.id,
                    name=mid,
                    capabilities=["chat", "streaming", "vision"],
                    context_length=int(m.get("inputTokenLimit", 32768) or 32768),
                    supports_streaming=True,
                    supports_tools=False,
                    supports_vision=True,
                    quality_score=float(self.config.extra.get("quality_score", 0.75)),
                    speed_score=float(self.config.extra.get("speed_score", 0.65)),
                    reliability_score=float(self.config.extra.get("reliability_score", 0.75)),
                    cost_score=float(self.config.extra.get("cost_score", 0.55)),
                )
            )
        return models

    async def health(self) -> ProviderHealth:
        if not self._configured:
            return ProviderHealth(status=ProviderStatus.DISABLED, last_checked=time.time(), last_error="not configured")
        start = time.time()
        try:
            resp = await self._client.get("/models", params={"key": self._api_key}, timeout=5.0)
            latency = (time.time() - start) * 1000
            if resp.status_code == 200:
                return ProviderHealth(status=ProviderStatus.HEALTHY, latency_ms=latency, last_checked=time.time())
            return ProviderHealth(
                status=ProviderStatus.DEGRADED, latency_ms=latency, last_checked=time.time(), last_error=f"HTTP {resp.status_code}"
            )
        except Exception as e:
            return ProviderHealth(status=ProviderStatus.OFFLINE, last_checked=time.time(), last_error=str(e))

    def _raise_for_status(self, resp: httpx.Response) -> None:
        if resp.status_code < 400:
            return
        text = resp.text[:300]
        if resp.status_code in (401, 403):
            raise ProviderAuthError(f"Auth failed ({resp.status_code}): {text}", provider_id=self.id)
        if resp.status_code == 429:
            raise ProviderRateLimitError(f"Rate limited: {text}", provider_id=self.id)
        if resp.status_code >= 500:
            raise ProviderServerError(f"HTTP {resp.status_code}: {text}", provider_id=self.id)
        raise ProviderError(f"HTTP {resp.status_code}: {text}", provider_id=self.id)

    async def chat(self, model: str, request: ChatCompletionRequest) -> ChatCompletionResponse:
        self._require_configured()
        self._usage.requests_total += 1
        contents, system_instruction = _to_gemini_contents(request)
        body: dict = {"contents": contents}
        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        if request.temperature is not None:
            body["generationConfig"] = {"temperature": request.temperature}

        try:
            resp = await self._client.post(f"/models/{model}:generateContent", params={"key": self._api_key}, json=body)
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
        candidates = data.get("candidates", [])
        text = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)
        usage_raw = data.get("usageMetadata", {}) or {}
        usage = Usage(
            prompt_tokens=usage_raw.get("promptTokenCount", 0),
            completion_tokens=usage_raw.get("candidatesTokenCount", 0),
            total_tokens=usage_raw.get("totalTokenCount", 0),
        )
        self._usage.tokens_total += usage.total_tokens
        return ChatCompletionResponse(
            model=f"{self.id}/{model}",
            choices=[ChatCompletionChoice(index=0, message={"role": "assistant", "content": text}, finish_reason="stop")],
            usage=usage,
        )

    async def stream_chat(self, model: str, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionChunk]:
        self._require_configured()
        self._usage.requests_total += 1
        contents, system_instruction = _to_gemini_contents(request)
        body: dict = {"contents": contents}
        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        chunk_id = f"chatcmpl-{int(time.time()*1000)}"

        try:
            async with self._client.stream(
                "POST",
                f"/models/{model}:streamGenerateContent",
                params={"key": self._api_key, "alt": "sse"},
                json=body,
            ) as resp:
                if resp.status_code >= 400:
                    self._usage.requests_failed += 1
                    await resp.aread()
                    self._raise_for_status(resp)

                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:"):].strip()
                    if not payload:
                        continue
                    try:
                        data = _json.loads(payload)
                    except _json.JSONDecodeError:
                        continue
                    candidates = data.get("candidates", [])
                    if not candidates:
                        continue
                    parts = candidates[0].get("content", {}).get("parts", [])
                    text = "".join(p.get("text", "") for p in parts)
                    finish_reason = "stop" if candidates[0].get("finishReason") else None
                    yield ChatCompletionChunk(
                        id=chunk_id,
                        model=f"{self.id}/{model}",
                        choices=[ChatCompletionChunkChoice(index=0, delta={"content": text}, finish_reason=finish_reason)],
                    )
        except httpx.TimeoutException as e:
            self._usage.requests_failed += 1
            raise ProviderTimeoutError(str(e), provider_id=self.id) from e
        except httpx.ConnectError as e:
            self._usage.requests_failed += 1
            raise ProviderConnectionError(str(e), provider_id=self.id) from e

    async def close(self) -> None:
        await self._client.aclose()
