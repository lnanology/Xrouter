"""The orchestration entrypoint used by the API layer: cache check -> route
-> execute (with fallback) -> telemetry. This is the "Fast Path" of section
六 — there is no agent orchestration in Phase 1, so every request takes this
same straight-line path regardless of classified complexity; complexity is
recorded for telemetry/future use only."""
from __future__ import annotations

import time
from collections.abc import AsyncIterator

from app.contracts.request import ChatCompletionRequest
from app.contracts.response import ChatCompletionChunk, ChatCompletionResponse
from app.core.context import AppContext
from app.core.errors import NoAvailableModelError
from app.execution.race import run_race
from app.intelligence.task_classifier import TaskClassification, classify
from app.observability.logging import get_logger
from app.routing.fallback import run_chat, run_stream_chat
from app.utils.ids import new_id
from app.storage.repositories.request import RequestRecord

logger = get_logger("engine")


class ChatEngine:
    def __init__(self, ctx: AppContext):
        self.ctx = ctx

    def _resolve_policy(self, request: ChatCompletionRequest, classification: TaskClassification) -> str:
        """A client-supplied routing_policy always wins. Otherwise, if the
        Task Classifier is enabled (section 三十六 Phase 2), use its
        suggestion; else fall back to the static Phase 1 default_policy."""
        if request.routing_policy:
            return request.routing_policy
        if self.ctx.settings.routing.task_aware_policy:
            return classification.suggested_policy
        return self.ctx.settings.routing.default_policy

    def _use_race(self, request: ChatCompletionRequest) -> bool:
        """Race mode (section 三十六) needs both the server-wide switch on
        and an explicit per-request opt-in, and only ever applies to
        non-streaming requests — see app/execution/race.py for why."""
        return bool(self.ctx.settings.routing.race_mode_enabled and request.race and not request.stream)

    async def handle_chat(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        request_id = new_id("req")
        classification = classify(request)
        policy = self._resolve_policy(request, classification)
        record = RequestRecord(
            request_id=request_id, task_type=classification.task_type.value,
            complexity=classification.complexity, routing_policy=policy,
        )
        start = time.time()

        cache_key = None
        if self.ctx.cache.is_cacheable(request):
            cache_key = self.ctx.cache.key_for(request)
            cached = await self.ctx.cache.get(cache_key)
            if cached is not None:
                record.status = "success"
                record.cache_hit = True
                record.latency_ms = (time.time() - start) * 1000
                await self.ctx.request_repo.save(record)
                self.ctx.metrics.record_request(latency_ms=record.latency_ms, success=True, cache_hit=True)
                self.ctx.events.emit("cache.hit", {"request_id": request_id})
                return ChatCompletionResponse.model_validate(cached)
            self.ctx.events.emit("cache.miss", {"request_id": request_id})

        try:
            decision = self.ctx.router.select(request, policy)
        except NoAvailableModelError as e:
            record.status = "failed"
            record.error = str(e)
            record.latency_ms = (time.time() - start) * 1000
            await self.ctx.request_repo.save(record)
            self.ctx.metrics.record_request(latency_ms=record.latency_ms, success=False)
            raise

        self.ctx.events.emit("provider.selected", {"request_id": request_id, "provider_id": decision.primary.provider_id})

        attempts_seen: list[dict] = []

        def _on_attempt(attempt: dict) -> None:
            attempts_seen.append(attempt)

        try:
            if self._use_race(request):
                response, attempts = await run_race(
                    decision, self.ctx.providers, self.ctx.circuits, self.ctx.quota, self.ctx.limiter,
                    request, self.ctx.settings.routing.max_fallback_attempts,
                    self.ctx.settings.routing.race_candidate_count,
                    events=self.ctx.events, performance=self.ctx.performance,
                )
                for a in attempts:
                    _on_attempt(a)
            else:
                response, attempts = await run_chat(
                    decision, self.ctx.providers, self.ctx.circuits, self.ctx.quota, self.ctx.limiter,
                    request, self.ctx.settings.routing.max_fallback_attempts, on_attempt=_on_attempt,
                    performance=self.ctx.performance,
                )
        except NoAvailableModelError as e:
            record.status = "failed"
            record.error = str(e)
            record.fallback_count = len(e.attempts)
            record.latency_ms = (time.time() - start) * 1000
            await self.ctx.request_repo.save(record)
            self.ctx.metrics.record_request(latency_ms=record.latency_ms, success=False)
            for i, a in enumerate(e.attempts):
                await self.ctx.request_repo.record_attempt(
                    request_id, i, a["provider_id"], a["model_id"], a["status"], a.get("latency_ms"), a.get("error")
                )
            raise

        latency_ms = (time.time() - start) * 1000
        record.status = "success"
        record.latency_ms = latency_ms
        record.tokens_prompt = response.usage.prompt_tokens
        record.tokens_completion = response.usage.completion_tokens
        record.fallback_count = max(len(attempts) - 1, 0)
        for i, a in enumerate(attempts):
            await self.ctx.request_repo.record_attempt(
                request_id, i, a["provider_id"], a["model_id"], a["status"], a.get("latency_ms"), a.get("error")
            )
        await self.ctx.request_repo.save(record)
        self.ctx.metrics.record_request(
            latency_ms=latency_ms, tokens=response.usage.total_tokens, success=True,
            fallback=record.fallback_count > 0, cache_hit=False,
        )
        self.ctx.events.emit("request.completed", {"request_id": request_id, "latency_ms": latency_ms})

        if response.xrouter:
            response.xrouter["request_id"] = request_id

        if cache_key is not None:
            await self.ctx.cache.set(cache_key, response.model_dump())

        return response

    async def handle_stream_chat(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionChunk]:
        request_id = new_id("req")
        classification = classify(request)
        policy = self._resolve_policy(request, classification)
        record = RequestRecord(
            request_id=request_id, task_type=classification.task_type.value,
            complexity=classification.complexity, routing_policy=policy,
        )
        start = time.time()
        first_token_at: float | None = None
        total_chunks = 0

        try:
            decision = self.ctx.router.select(request, policy)
        except NoAvailableModelError as e:
            record.status = "failed"
            record.error = str(e)
            record.latency_ms = (time.time() - start) * 1000
            await self.ctx.request_repo.save(record)
            self.ctx.metrics.record_request(latency_ms=record.latency_ms, success=False)
            raise

        self.ctx.events.emit("provider.selected", {"request_id": request_id, "provider_id": decision.primary.provider_id})
        attempts_seen: list[dict] = []

        def _on_attempt(attempt: dict) -> None:
            attempts_seen.append(attempt)

        try:
            async for chunk in run_stream_chat(
                decision, self.ctx.providers, self.ctx.circuits, self.ctx.quota, self.ctx.limiter,
                request, self.ctx.settings.routing.max_fallback_attempts, on_attempt=_on_attempt,
                performance=self.ctx.performance,
            ):
                if first_token_at is None:
                    first_token_at = time.time()
                total_chunks += 1
                yield chunk
        except NoAvailableModelError as e:
            record.status = "failed"
            record.error = str(e)
            record.fallback_count = len(e.attempts)
            record.latency_ms = (time.time() - start) * 1000
            await self.ctx.request_repo.save(record)
            self.ctx.metrics.record_request(latency_ms=record.latency_ms, success=False, timeout="timed out" in str(e).lower())
            raise
        finally:
            latency_ms = (time.time() - start) * 1000
            ttft_ms = (first_token_at - start) * 1000 if first_token_at else None
            if record.status != "failed":
                record.status = "success"
                record.latency_ms = latency_ms
                record.ttft_ms = ttft_ms
                record.fallback_count = max(len(attempts_seen) - 1, 0)
                for i, a in enumerate(attempts_seen):
                    await self.ctx.request_repo.record_attempt(
                        request_id, i, a["provider_id"], a["model_id"], a["status"], a.get("latency_ms"), a.get("error")
                    )
                await self.ctx.request_repo.save(record)
                tps = (total_chunks / (latency_ms / 1000)) if latency_ms > 0 else None
                self.ctx.metrics.record_request(latency_ms=latency_ms, ttft_ms=ttft_ms, tokens_per_sec=tps, success=True)
                self.ctx.events.emit("request.completed", {"request_id": request_id, "latency_ms": latency_ms})
