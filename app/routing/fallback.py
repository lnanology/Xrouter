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


async def try_candidate(
    idx: int,
    candidate: RoutingCandidate,
    providers: ProviderRegistry,
    circuits: CircuitBreakerRegistry,
    quota: QuotaTracker,
    limiter: ConcurrencyLimiter,
    request: ChatCompletionRequest,
    performance: PerformanceController | None = None,
) -> tuple[ChatCompletionResponse | None, dict]:
    """Runs a single candidate attempt end-to-end (breaker admission ->
    concurrency admission -> bounded timeout -> bounded retry -> breaker/
    quota/performance bookkeeping) and never raises ProviderError itself —
    it returns (None, attempt_log_entry) on any failure instead of raising,
    so both the sequential fallback chain (run_chat, below) and race mode
    (app/execution/race.py) can share one implementation of this bookkeeping
    without either having to catch the other's exceptions."""
    provider = providers.get(candidate.provider_id)
    breaker = circuits.get(candidate.provider_id)
    if provider is None or not breaker.allow_request():
        return None, {"provider_id": candidate.provider_id, "model_id": candidate.model_id, "status": "skipped"}

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
            "attempt_index": idx,
            "latency_ms": round(latency_ms, 1),
        }
        return response, {"provider_id": candidate.provider_id, "model_id": candidate.model_id, "status": "success", "latency_ms": latency_ms}
    except ProviderError as e:
        latency_ms = (time.time() - start) * 1000
        breaker.record_failure()
        if isinstance(e, ProviderRateLimitError):
            quota.record_rate_limit(candidate.provider_id, e.retry_after)
        if performance is not None:
            performance.record(candidate.provider_id, latency_ms, success=False)
        return None, {
            "provider_id": candidate.provider_id, "model_id": candidate.model_id, "status": "failed",
            "latency_ms": latency_ms, "error": str(e),
        }


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
        response, attempt = await try_candidate(idx, candidate, providers, circuits, quota, limiter, request, performance)
        attempts_log.append(attempt)
        if on_attempt:
            on_attempt(attempt)
        if response is not None:
            response.xrouter["policy"] = decision.policy
            return response, attempts_log

    raise NoAvailableModelError("All candidates in the fallback chain failed.", attempts=attempts_log)


async def try_candidate_stream(
    idx: int,
    candidate: RoutingCandidate,
    providers: ProviderRegistry,
    circuits: CircuitBreakerRegistry,
    quota: QuotaTracker,
    limiter: ConcurrencyLimiter,
    request: ChatCompletionRequest,
    performance: PerformanceController | None = None,
) -> tuple[AsyncIterator[ChatCompletionChunk] | None, ChatCompletionChunk | None, dict]:
    """Streaming counterpart to try_candidate (above): runs one candidate's
    admission -> bounded-timeout wait for its FIRST chunk only, then hands
    the still-open generator back to the caller rather than draining it
    itself — the same "commit on first chunk" boundary run_stream_chat has
    always used, just factored out so both the sequential fallback
    (run_stream_chat, below) and streaming race mode
    (app/execution/race.py's run_stream_race) can share it, mirroring how
    try_candidate is already shared between run_chat and run_race.

    Returns (generator, first_chunk, attempt) on success — the caller owns
    draining (and, if it ends up not being used, aclose()-ing) the
    generator from here on. Returns (None, None, attempt) on skip/failure.
    Never raises."""
    provider = providers.get(candidate.provider_id)
    breaker = circuits.get(candidate.provider_id)
    if provider is None or not breaker.allow_request():
        return None, None, {"provider_id": candidate.provider_id, "model_id": candidate.model_id, "status": "skipped"}

    start = time.time()
    quota.record_request(candidate.provider_id)
    gen = provider.stream_chat(candidate.model_id, request)
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
        return None, None, {
            "provider_id": candidate.provider_id, "model_id": candidate.model_id, "status": "failed", "error": str(e),
        }

    # First chunk arrived (or the stream ended immediately, empty): commit.
    breaker.record_success()
    quota.record_success(candidate.provider_id)
    latency_ms = (time.time() - start) * 1000
    if performance is not None:
        performance.record(candidate.provider_id, latency_ms, success=True)
    attempt = {
        "provider_id": candidate.provider_id, "model_id": candidate.model_id, "status": "success", "latency_ms": latency_ms,
    }
    return gen, first_chunk, attempt


async def drain_stream(
    candidate: RoutingCandidate, gen: AsyncIterator[ChatCompletionChunk], first_chunk: ChatCompletionChunk | None,
) -> AsyncIterator[ChatCompletionChunk]:
    """Shared tail end of streaming a committed candidate: yield the first
    chunk (already consumed by try_candidate_stream) then the rest of the
    generator, turning a mid-stream ProviderError into a terminal in-band
    error chunk instead of a raised exception (the client already saw
    partial output from this specific model, so failing over silently is
    not possible anymore)."""
    if first_chunk is not None:
        yield first_chunk
    try:
        async for chunk in gen:
            yield chunk
    except ProviderError as e:
        yield ChatCompletionChunk(
            id="chatcmpl-error",
            model=f"{candidate.provider_id}/{candidate.model_id}",
            choices=[{"index": 0, "delta": {"content": f"\n[xrouter: stream interrupted: {e}]"}, "finish_reason": "error"}],
        )


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
        gen, first_chunk, attempt = await try_candidate_stream(
            idx, candidate, providers, circuits, quota, limiter, request, performance,
        )
        if attempt["status"] == "skipped":
            # Preserves this function's existing behavior exactly: a
            # breaker-blocked/missing-provider candidate was never logged
            # here (unlike run_chat/run_race, which do log skips) — the
            # `continue` used to happen before any attempt dict existed at
            # all, so it still isn't recorded now that one does.
            continue
        attempts_log.append(attempt)
        if on_attempt:
            on_attempt(attempt)
        if gen is None:
            last_error = RuntimeError(attempt.get("error", "stream failed to start"))
            continue

        async for chunk in drain_stream(candidate, gen, first_chunk):
            yield chunk
        return

    raise NoAvailableModelError(
        f"All candidates failed to start streaming: {last_error}", attempts=attempts_log
    )
