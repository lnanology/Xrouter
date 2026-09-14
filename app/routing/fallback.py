"""Executes a RoutingDecision's candidate chain in order (section 三十二):

    Provider A (retry on transient errors)
      -> Provider B
        -> ... -> Local (if configured)
          -> structured error if everything fails

Circuit breaker state and quota tracking are updated here, right where
success/failure is observed. Never retries indefinitely: bounded by
RetryConfig per attempt and by the length of the fallback chain overall.
"""
from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from typing import Any

from app.contracts.request import ChatCompletionRequest
from app.contracts.response import ChatCompletionChunk, ChatCompletionResponse
from app.contracts.router import RoutingCandidate, RoutingDecision
from app.core.errors import NoAvailableModelError, ProviderError, ProviderRateLimitError
from app.core.registry import ProviderRegistry
from app.execution.async_executor import execute_attempt
from app.quota.tracker import QuotaTracker
from app.reliability.circuit_breaker import CircuitBreakerRegistry
from app.reliability.retry import RetryConfig, retry_with_backoff
from app.routing.performance_controller import PerformanceController
from app.routing.scheduler import ConcurrencyLimiter

AttemptHook = Callable[[dict[str, Any]], None]


def _candidates(decision: RoutingDecision) -> list[RoutingCandidate]:
    return [decision.primary, *decision.fallback_chain]


async def run_chat(
    decision: RoutingDecision,
    providers: ProviderRegistry,
    circuits: CircuitBreakerRegistry,
    quota: QuotaTracker,
    limiter: ConcurrencyLimiter,
    request: ChatCompletionRequest,
    max_attempts: int,
    on_attempt: AttemptHook | None = None,
    performance: PerformanceController | None = None,
) -> tuple[ChatCompletionResponse, list[dict]]:
    attempts_log: list[dict] = []
    for idx, candidate in enumerate(_candidates(decision)[:max_attempts]):
        provider = providers.get(candidate.provider_id)
        breaker = circuits.get(candidate.provider_id)
        if provider is None or not breaker.allow_request():
            attempts_log.append({"provider_id": candidate.provider_id, "model_id": candidate.model_id, "status": "skipped"})
            continue

        start = time.time()
        quota.record_request(candidate.provider_id)
        try:
            async def _call():
                return await provider.chat(candidate.model_id, request)

            response = await retry_with_backoff(
                lambda: execute_attempt(candidate.provider_id, limiter, provider.config.timeout_seconds, _call),
                RetryConfig(),
            )
            latency_ms = (time.time() - start) * 1000
            breaker.record_success()
            quota.record_success(candidate.provider_id)
            quota.record_tokens(candidate.provider_id, response.usage.total_tokens)
            if performance is not None:
                performance.record(candidate.provider_id, latency_ms, success=True)
            response.xrouter = {
                "provider": candidate.provider_id,
                "model": candidate.model_id,
                "policy": decision.policy,
                "attempt_index": idx,
                "latency_ms": round(latency_ms, 1),
            }
            attempt = {"provider_id": candidate.provider_id, "model_id": candidate.model_id, "status": "success", "latency_ms": latency_ms}
            attempts_log.append(attempt)
            if on_attempt:
                on_attempt(attempt)
            return response, attempts_log
        except ProviderError as e:
            latency_ms = (time.time() - start) * 1000
            breaker.record_failure()
            if isinstance(e, ProviderRateLimitError):
                quota.record_rate_limit(candidate.provider_id, e.retry_after)
            if performance is not None:
                performance.record(candidate.provider_id, latency_ms, success=False)
            attempt = {
                "provider_id": candidate.provider_id, "model_id": candidate.model_id, "status": "failed",
                "latency_ms": latency_ms, "error": str(e),
            }
            attempts_log.append(attempt)
            if on_attempt:
                on_attempt(attempt)
            continue

    raise NoAvailableModelError("All candidates in the fallback chain failed.", attempts=attempts_log)


async def run_stream_chat(
    decision: RoutingDecision,
    providers: ProviderRegistry,
    circuits: CircuitBreakerRegistry,
    quota: QuotaTracker,
    limiter: ConcurrencyLimiter,
    request: ChatCompletionRequest,
    max_attempts: int,
    on_attempt: AttemptHook | None = None,
    performance: PerformanceController | None = None,
) -> AsyncIterator[ChatCompletionChunk]:
    """Falls back to the next candidate only if the *first* chunk fails to
    arrive. Once streaming has started and content has been sent to the
    client, a mid-stream failure cannot be silently retried on another
    model (the client already saw partial output from a specific model) —
    it is surfaced as a terminal error instead."""
    attempts_log: list[dict] = []
    last_error: Exception | None = None

    for idx, candidate in enumerate(_candidates(decision)[:max_attempts]):
        provider = providers.get(candidate.provider_id)
        breaker = circuits.get(candidate.provider_id)
        if provider is None or not breaker.allow_request():
            continue

        start = time.time()
        quota.record_request(candidate.provider_id)
        gen = provider.stream_chat(candidate.model_id, request)
        first_chunk: ChatCompletionChunk | None = None
        try:
            async def _first():
                return await gen.__anext__()

            first_chunk = await execute_attempt(candidate.provider_id, limiter, provider.config.timeout_seconds, _first)
        except StopAsyncIteration:
            first_chunk = None
        except ProviderError as e:
            breaker.record_failure()
            if isinstance(e, ProviderRateLimitError):
                quota.record_rate_limit(candidate.provider_id, e.retry_after)
            if performance is not None:
                performance.record(candidate.provider_id, (time.time() - start) * 1000, success=False)
            last_error = e
            attempt = {"provider_id": candidate.provider_id, "model_id": candidate.model_id, "status": "failed", "error": str(e)}
            attempts_log.append(attempt)
            if on_attempt:
                on_attempt(attempt)
            continue

        # First chunk arrived: commit to this provider for the rest of the stream.
        breaker.record_success()
        quota.record_success(candidate.provider_id)
        first_chunk_latency_ms = (time.time() - start) * 1000
        if performance is not None:
            performance.record(candidate.provider_id, first_chunk_latency_ms, success=True)
        attempt = {
            "provider_id": candidate.provider_id, "model_id": candidate.model_id, "status": "success",
            "latency_ms": first_chunk_latency_ms,
        }
        attempts_log.append(attempt)
        if on_attempt:
            on_attempt(attempt)

        if first_chunk is not None:
            yield first_chunk
        try:
            async for chunk in gen:
                yield chunk
        except ProviderError as e:
            # Mid-stream failure: cannot fail over anymore, surface as final chunk.
            yield ChatCompletionChunk(
                id="chatcmpl-error",
                model=f"{candidate.provider_id}/{candidate.model_id}",
                choices=[{"index": 0, "delta": {"content": f"\n[xrouter: stream interrupted: {e}]"}, "finish_reason": "error"}],
            )
        return

    raise NoAvailableModelError(
        f"All candidates failed to start streaming: {last_error}", attempts=attempts_log
    )
