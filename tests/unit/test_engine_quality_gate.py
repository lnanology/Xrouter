"""ChatEngine._apply_quality_gate is exercised directly (not through the
full handle_chat/cache/DB pipeline — that's what tests/integration/test_api
covers) with a real ProviderRegistry + FakeProvider pair standing in for
"a degenerate local model" and "a good fallback model", the same pattern
tests/execution/test_race.py uses for run_race."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.contracts.provider import ProviderConfig
from app.contracts.request import ChatCompletionRequest, ChatMessage
from app.contracts.router import RoutingCandidate, RoutingDecision
from app.core.engine import ChatEngine
from app.core.registry import ProviderRegistry
from app.observability.events import EventBus
from app.quota.tracker import QuotaTracker
from app.reliability.circuit_breaker import CircuitBreakerRegistry
from app.routing.fallback import run_chat
from app.routing.scheduler import ConcurrencyLimiter
from tests.helpers import FakeProvider, make_model


def _engine(min_score=0.5, max_retries=1):
    providers = ProviderRegistry()
    circuits = CircuitBreakerRegistry()
    quota = QuotaTracker()
    limiter = ConcurrencyLimiter(global_limit=16)
    events = EventBus()
    ctx = SimpleNamespace(
        settings=SimpleNamespace(routing=SimpleNamespace(quality_gate_min_score=min_score, max_quality_retries=max_retries)),
        providers=providers, circuits=circuits, quota=quota, limiter=limiter, events=events, performance=None,
    )
    return ChatEngine(ctx), providers, limiter


def _register(providers, limiter, pid, **kwargs):
    cfg = ProviderConfig(id=pid, name=pid, type="openai_compatible", timeout_seconds=2.0)
    fake = FakeProvider(cfg, [make_model(pid)], **kwargs)
    providers.register(fake, cfg)
    limiter.configure_provider(pid, 4)
    return fake


def _req():
    return ChatCompletionRequest(model="auto", messages=[ChatMessage(role="user", content="hi")])


@pytest.mark.asyncio
async def test_passing_response_is_not_retried():
    engine, providers, limiter = _engine()
    _register(providers, limiter, "good", content="This is a solid, complete answer.")
    decision = RoutingDecision(primary=RoutingCandidate(provider_id="good", model_id="test-model", score=1.0))
    response, attempts = await run_chat(decision, providers, engine.ctx.circuits, engine.ctx.quota, limiter, _req(), max_attempts=1)

    final, final_attempts = await engine._apply_quality_gate(decision, _req(), response, attempts)

    assert final.xrouter["quality"]["passed"] is True
    assert "quality_retries" not in final.xrouter
    assert len(final_attempts) == 1


@pytest.mark.asyncio
async def test_degenerate_response_retries_to_next_untried_candidate():
    engine, providers, limiter = _engine()
    _register(providers, limiter, "bad", content="the the the the the the the the the the")
    _register(providers, limiter, "good", content="This is a solid, complete answer.")
    decision = RoutingDecision(
        primary=RoutingCandidate(provider_id="bad", model_id="test-model", score=1.0),
        fallback_chain=[RoutingCandidate(provider_id="good", model_id="test-model", score=0.9)],
    )
    first_hop = RoutingDecision(primary=decision.primary)
    response, attempts = await run_chat(first_hop, providers, engine.ctx.circuits, engine.ctx.quota, limiter, _req(), max_attempts=1)
    assert response.xrouter["provider"] == "bad"

    final, final_attempts = await engine._apply_quality_gate(decision, _req(), response, attempts)

    assert final.xrouter["provider"] == "good"
    assert final.xrouter["quality"]["passed"] is True
    assert final.xrouter["quality_retries"] == 1
    assert [a["status"] for a in final_attempts] == ["success", "success"]


@pytest.mark.asyncio
async def test_all_candidates_degenerate_returns_best_effort_without_raising():
    engine, providers, limiter = _engine()
    _register(providers, limiter, "bad1", content="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    _register(providers, limiter, "bad2", content="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
    decision = RoutingDecision(
        primary=RoutingCandidate(provider_id="bad1", model_id="test-model", score=1.0),
        fallback_chain=[RoutingCandidate(provider_id="bad2", model_id="test-model", score=0.9)],
    )
    first_hop = RoutingDecision(primary=decision.primary)
    response, attempts = await run_chat(first_hop, providers, engine.ctx.circuits, engine.ctx.quota, limiter, _req(), max_attempts=1)

    final, final_attempts = await engine._apply_quality_gate(decision, _req(), response, attempts)

    # Never raises — returns the last attempted (still-degenerate) response
    # rather than erroring out, per XRouter's graceful-degradation rule.
    assert final is not None
    assert final.xrouter["quality"]["passed"] is False
    assert final.xrouter["quality_retries"] == 1
    assert len(final_attempts) == 2


@pytest.mark.asyncio
async def test_max_retries_zero_assesses_but_never_retries():
    engine, providers, limiter = _engine(max_retries=0)
    _register(providers, limiter, "bad", content="the the the the the the the the the the")
    _register(providers, limiter, "good", content="This is a solid, complete answer.")
    decision = RoutingDecision(
        primary=RoutingCandidate(provider_id="bad", model_id="test-model", score=1.0),
        fallback_chain=[RoutingCandidate(provider_id="good", model_id="test-model", score=0.9)],
    )
    first_hop = RoutingDecision(primary=decision.primary)
    response, attempts = await run_chat(first_hop, providers, engine.ctx.circuits, engine.ctx.quota, limiter, _req(), max_attempts=1)

    final, final_attempts = await engine._apply_quality_gate(decision, _req(), response, attempts)

    assert final.xrouter["provider"] == "bad"
    assert final.xrouter["quality"]["passed"] is False
    assert "quality_retries" not in final.xrouter
    assert len(final_attempts) == 1


@pytest.mark.asyncio
async def test_quality_failed_event_emitted_on_gate_failure():
    engine, providers, limiter = _engine()
    _register(providers, limiter, "bad", content="the the the the the the the the the the")
    _register(providers, limiter, "good", content="This is a solid, complete answer.")
    seen = []
    engine.ctx.events.subscribe("quality.failed", lambda p: seen.append(p))
    decision = RoutingDecision(
        primary=RoutingCandidate(provider_id="bad", model_id="test-model", score=1.0),
        fallback_chain=[RoutingCandidate(provider_id="good", model_id="test-model", score=0.9)],
    )
    first_hop = RoutingDecision(primary=decision.primary)
    response, attempts = await run_chat(first_hop, providers, engine.ctx.circuits, engine.ctx.quota, limiter, _req(), max_attempts=1)

    await engine._apply_quality_gate(decision, _req(), response, attempts)

    assert len(seen) == 1
    assert "degenerate_word_repetition" in seen[0]["reasons"]
