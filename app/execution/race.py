"""Race mode (Phase 2, spec section 三十六): dispatch the top-N ranked
candidates concurrently and return whichever succeeds first, cancelling
whatever's still in flight. Reuses the exact same per-candidate breaker/
quota/retry/performance bookkeeping as the sequential fallback chain via
app.routing.fallback.try_candidate, so racing never bypasses circuit
breakers, quota tracking, or bounded retry — it only changes *when* those
per-candidate attempts are dispatched (concurrently vs. one-at-a-time).

Scope for this phase: non-streaming chat completions only. Racing partial
token streams (buffering/discarding, cancelling mid-stream) is a
substantially different problem — left for later, see README "not
implemented yet".

Opt-in per request (`"race": true` in the request body) and gated by the
server-wide `routing.race_mode_enabled` switch, which defaults to off so a
request never silently starts consuming quota on multiple providers unless
an operator has explicitly turned racing on."""
from __future__ import annotations

import asyncio

from app.contracts.request import ChatCompletionRequest
from app.contracts.response import ChatCompletionResponse
from app.contracts.router import RoutingCandidate, RoutingDecision
from app.core.errors import NoAvailableModelError
from app.core.registry import ProviderRegistry
from app.observability.events import EventBus
from app.quota.tracker import QuotaTracker
from app.reliability.circuit_breaker import CircuitBreakerRegistry
from app.routing.fallback import try_candidate
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
