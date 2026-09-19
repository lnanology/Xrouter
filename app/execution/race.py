"""Race mode (Phase 2, spec section 三十六): dispatch the top-N ranked
candidates concurrently and return whichever succeeds first, cancelling
whatever's still in flight. Reuses the exact same per-candidate breaker/
quota/retry/performance bookkeeping as the sequential fallback chain via
app.routing.fallback.try_candidate, so racing never bypasses circuit
breakers, quota tracking, or bounded retry — it only changes *when* those
per-candidate attempts are dispatched (concurrently vs. one-at-a-time).

Streaming requests are raced too, via run_stream_race (below) — it races
on *first-chunk arrival* rather than a full response, reusing
app.routing.fallback.try_candidate_stream (the streaming counterpart to
try_candidate) the same way run_race reuses try_candidate. One concern
non-streaming racing never had: a losing candidate that already received
its first chunk holds an open provider-side stream nothing will ever
drain — run_stream_race explicitly aclose()s every such generator so it
never leaks a connection.

Opt-in per request (`"race": true` in the request body) and gated by the
server-wide `routing.race_mode_enabled` switch, which defaults to off so a
request never silently starts consuming quota on multiple providers unless
an operator has explicitly turned racing on."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from app.contracts.request import ChatCompletionRequest
from app.contracts.response import ChatCompletionChunk, ChatCompletionResponse
from app.contracts.router import RoutingCandidate, RoutingDecision
from app.core.errors import NoAvailableModelError
from app.core.registry import ProviderRegistry
from app.observability.events import EventBus
from app.quota.tracker import QuotaTracker
from app.reliability.circuit_breaker import CircuitBreakerRegistry
from app.routing.fallback import AttemptHook, drain_stream, try_candidate, try_candidate_stream
from app.routing.performance_controller import PerformanceController
from app.routing.scheduler import ConcurrencyLimiter


async def run_race(
    decision: RoutingDecision,
    providers: ProviderRegistry,
    circuits: CircuitBreakerRegistry,
    quota: QuotaTracker,
    limiter: ConcurrencyLimiter,
    request: ChatCompletionRequest,
    max_attempts: int,
    race_candidate_count: int,
    events: EventBus | None = None,
    performance: PerformanceController | None = None,
) -> tuple[ChatCompletionResponse, list[dict]]:
    all_candidates: list[RoutingCandidate] = [decision.primary, *decision.fallback_chain][:max_attempts]
    race_n = max(1, min(race_candidate_count, len(all_candidates)))
    raced, rest = all_candidates[:race_n], all_candidates[race_n:]

    attempts_log: list[dict] = []
    if events:
        events.emit("race.started", {"candidates": [c.provider_id for c in raced]})

    tasks: dict[asyncio.Task, RoutingCandidate] = {
        asyncio.ensure_future(try_candidate(idx, c, providers, circuits, quota, limiter, request, performance)): c
        for idx, c in enumerate(raced)
    }

    winner: ChatCompletionResponse | None = None
    winner_candidate: RoutingCandidate | None = None
    pending = set(tasks.keys())
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            response, attempt = task.result()
            attempts_log.append(attempt)
            if response is not None and winner is None:
                winner = response
                winner_candidate = tasks[task]
        if winner is not None:
            break

    if pending:
        # Either we already have a winner, or every raced candidate has
        # reported in and failed — either way, nothing left in `pending`
        # should keep running.
        cancelled = [tasks[t] for t in pending]
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for c in cancelled:
            attempts_log.append({"provider_id": c.provider_id, "model_id": c.model_id, "status": "cancelled"})

    if winner is not None:
        if events:
            events.emit("race.completed", {"winner": winner_candidate.provider_id})
        winner.xrouter["policy"] = decision.policy
        winner.xrouter["race"] = True
        winner.xrouter["race_candidates"] = len(raced)
        return winner, attempts_log

    # Every raced candidate failed — fall back to the remaining candidates
    # sequentially, same semantics as the non-race fallback chain.
    for idx, candidate in enumerate(rest, start=len(raced)):
        response, attempt = await try_candidate(idx, candidate, providers, circuits, quota, limiter, request, performance)
        attempts_log.append(attempt)
        if response is not None:
            response.xrouter["policy"] = decision.policy
            response.xrouter["race"] = True
            response.xrouter["race_candidates"] = len(raced)
            return response, attempts_log

    raise NoAvailableModelError("All raced and fallback candidates failed.", attempts=attempts_log)


async def run_stream_race(
    decision: RoutingDecision,
    providers: ProviderRegistry,
    circuits: CircuitBreakerRegistry,
    quota: QuotaTracker,
    limiter: ConcurrencyLimiter,
    request: ChatCompletionRequest,
    max_attempts: int,
    race_candidate_count: int,
    on_attempt: AttemptHook | None = None,
    events: EventBus | None = None,
    performance: PerformanceController | None = None,
) -> AsyncIterator[ChatCompletionChunk]:
    """Streaming counterpart to run_race: same dispatch/cancel/fallback
    shape, but races on first-chunk arrival (via try_candidate_stream)
    instead of a full response, and yields chunks rather than returning a
    value. Attempts are both accumulated locally (for a NoAvailableModelError
    on total failure, mirroring run_race's own contract) and reported
    through on_attempt as they happen, the same callback pattern
    app.routing.fallback.run_stream_chat already uses for its own
    (response-less) async-generator return type."""
    all_candidates: list[RoutingCandidate] = [decision.primary, *decision.fallback_chain][:max_attempts]
    race_n = max(1, min(race_candidate_count, len(all_candidates)))
    raced, rest = all_candidates[:race_n], all_candidates[race_n:]

    attempts_log: list[dict] = []

    def _log(attempt: dict) -> None:
        attempts_log.append(attempt)
        if on_attempt:
            on_attempt(attempt)

    if events:
        events.emit("race.started", {"candidates": [c.provider_id for c in raced]})

    tasks: dict[asyncio.Task, RoutingCandidate] = {
        asyncio.ensure_future(
            try_candidate_stream(idx, c, providers, circuits, quota, limiter, request, performance)
        ): c
        for idx, c in enumerate(raced)
    }

    winner_gen: AsyncIterator[ChatCompletionChunk] | None = None
    winner_first_chunk: ChatCompletionChunk | None = None
    winner_candidate: RoutingCandidate | None = None
    pending = set(tasks.keys())
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            gen, first_chunk, attempt = task.result()
            _log(attempt)
            if gen is None:
                continue
            if winner_gen is None:
                winner_gen = gen
                winner_first_chunk = first_chunk
                winner_candidate = tasks[task]
            else:
                # This candidate also got its first chunk, but another one
                # already won the race in this same batch — its stream
                # will never be drained, so release it now.
                await gen.aclose()
        if winner_gen is not None:
            break

    if pending:
        # Either we already have a winner, or every raced candidate has
        # reported in and failed — either way, nothing left in `pending`
        # should keep running.
        pending_list = list(pending)
        cancelled = [tasks[t] for t in pending_list]
        for task in pending_list:
            task.cancel()
        results = await asyncio.gather(*pending_list, return_exceptions=True)
        for candidate, result in zip(cancelled, results):
            # A task can finish (with an already-open generator) in the
            # narrow window between asyncio.wait() returning above and the
            # cancel() call just above -- if so, its generator is real and
            # unused, so it needs closing too rather than being dropped.
            if isinstance(result, tuple):
                gen, _first_chunk, _attempt = result
                if gen is not None:
                    await gen.aclose()
            _log({"provider_id": candidate.provider_id, "model_id": candidate.model_id, "status": "cancelled"})

    if winner_gen is not None:
        if events:
            events.emit("race.completed", {"winner": winner_candidate.provider_id})
        async for chunk in drain_stream(winner_candidate, winner_gen, winner_first_chunk):
            yield chunk
        return

    # Every raced candidate failed to start streaming — fall back to the
    # remaining candidates sequentially, same semantics as run_stream_chat.
    for idx, candidate in enumerate(rest, start=len(raced)):
        gen, first_chunk, attempt = await try_candidate_stream(
            idx, candidate, providers, circuits, quota, limiter, request, performance,
        )
        _log(attempt)
        if gen is not None:
            async for chunk in drain_stream(candidate, gen, first_chunk):
                yield chunk
            return

    raise NoAvailableModelError("All raced and fallback candidates failed to start streaming.", attempts=attempts_log)
