"""ChatEngine._apply_quality_gate is exercised directly (not through the
full handle_chat/cache/DB pipeline — that's what tests/integration/test_api
covers) with a real ProviderRegistry + FakeProvider pair standing in for
"a degenerate local model" and "a good fallback model", the same pattern
tests/execution/test_race.py uses for run_race. LLM-graded judgment (the
second, opt-in half of the gate) reuses the same FakeProvider double for
the judge itself, scripted via its tool_calls/responses params to return a
forced submit_quality_judgment call -- no new test double needed."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.contracts.provider import ProviderConfig
from app.contracts.request import ChatCompletionRequest, ChatMessage
from app.contracts.router import RoutingCandidate, RoutingDecision
from app.core.engine import ChatEngine
from app.core.errors import NoAvailableModelError
from app.core.registry import ProviderRegistry
from app.intelligence.quality_gate import LLM_QUALITY_TOOL_NAME
from app.observability.events import EventBus
from app.quota.tracker import QuotaTracker
from app.reliability.circuit_breaker import CircuitBreakerRegistry
from app.routing.fallback import run_chat
from app.routing.scheduler import ConcurrencyLimiter
from tests.helpers import FakeProvider, make_model


class _FakeRouter:
    """Stands in for AdaptiveRouter for the judge call only -- returns a
    fixed decision (or raises NoAvailableModelError) regardless of the
    judge request/policy passed in, so tests can control exactly which
    provider gets picked as the judge."""

    def __init__(self, decision: RoutingDecision | None = None, raises: bool = False):
        self._decision = decision
        self._raises = raises

    def select(self, request, policy_name=None):
        if self._raises:
            raise NoAvailableModelError("no judge candidate available")
        return self._decision


def _judgment_tool_call(satisfied: bool, feedback: str = ""):
    import json

    return [{
        "id": "call_1", "type": "function",
        "function": {"name": LLM_QUALITY_TOOL_NAME, "arguments": json.dumps({"satisfied": satisfied, "feedback": feedback})},
    }]


def _engine(min_score=0.5, max_retries=1, llm_grading_enabled=False, judge_policy="quality", router=None):
    providers = ProviderRegistry()
    circuits = CircuitBreakerRegistry()
    quota = QuotaTracker()
    limiter = ConcurrencyLimiter(global_limit=16)
    events = EventBus()
    ctx = SimpleNamespace(
        settings=SimpleNamespace(routing=SimpleNamespace(
            quality_gate_min_score=min_score, max_quality_retries=max_retries,
            quality_gate_llm_grading_enabled=llm_grading_enabled, quality_gate_judge_policy=judge_policy,
        )),
        providers=providers, circuits=circuits, quota=quota, limiter=limiter, events=events, performance=None,
        router=router,
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


# --- LLM-graded judgment (opt-in second half of the gate) -------------------

@pytest.mark.asyncio
async def test_llm_grading_disabled_never_calls_the_judge():
    # router.select would raise if it were ever called -- proving the judge
    # is never even attempted when quality_gate_llm_grading_enabled is off,
    # the existing (unchanged) behavior this whole feature must not disturb.
    engine, providers, limiter = _engine(llm_grading_enabled=False, router=_FakeRouter(raises=True))
    _register(providers, limiter, "good", content="This is a solid, complete answer.")
    decision = RoutingDecision(primary=RoutingCandidate(provider_id="good", model_id="test-model", score=1.0))
    response, attempts = await run_chat(decision, providers, engine.ctx.circuits, engine.ctx.quota, limiter, _req(), max_attempts=1)

    final, _ = await engine._apply_quality_gate(decision, _req(), response, attempts)

    assert final.xrouter["quality"]["passed"] is True
    assert "llm_feedback" not in final.xrouter["quality"]


@pytest.mark.asyncio
async def test_llm_grading_satisfied_leaves_a_structurally_good_response_alone():
    judge_decision = RoutingDecision(primary=RoutingCandidate(provider_id="judge", model_id="test-model", score=1.0))
    engine, providers, limiter = _engine(llm_grading_enabled=True, router=_FakeRouter(decision=judge_decision))
    _register(providers, limiter, "good", content="This is a solid, complete answer.")
    _register(providers, limiter, "judge", tool_calls=_judgment_tool_call(satisfied=True))
    decision = RoutingDecision(primary=RoutingCandidate(provider_id="good", model_id="test-model", score=1.0))
    response, attempts = await run_chat(decision, providers, engine.ctx.circuits, engine.ctx.quota, limiter, _req(), max_attempts=1)

    final, final_attempts = await engine._apply_quality_gate(decision, _req(), response, attempts)

    assert final.xrouter["provider"] == "good"
    assert final.xrouter["quality"]["passed"] is True
    assert "llm_feedback" not in final.xrouter["quality"]
    assert "quality_retries" not in final.xrouter
    assert len(final_attempts) == 1  # the judge call itself is never logged into the response's own attempts


@pytest.mark.asyncio
async def test_llm_grading_unsatisfied_retries_to_next_candidate():
    judge_decision = RoutingDecision(primary=RoutingCandidate(provider_id="judge", model_id="test-model", score=1.0))
    engine, providers, limiter = _engine(llm_grading_enabled=True, router=_FakeRouter(decision=judge_decision))
    _register(providers, limiter, "good1", content="A plausible-looking but actually wrong answer.")
    _register(providers, limiter, "good2", content="The genuinely correct answer.")
    # First judge call (of "good1"'s answer): unsatisfied. Second (of
    # "good2"'s, after retry): satisfied.
    _register(providers, limiter, "judge", responses=[
        {"tool_calls": _judgment_tool_call(satisfied=False, feedback="Factually wrong.")},
        {"tool_calls": _judgment_tool_call(satisfied=True)},
    ])
    decision = RoutingDecision(
        primary=RoutingCandidate(provider_id="good1", model_id="test-model", score=1.0),
        fallback_chain=[RoutingCandidate(provider_id="good2", model_id="test-model", score=0.9)],
    )
    first_hop = RoutingDecision(primary=decision.primary)
    response, attempts = await run_chat(first_hop, providers, engine.ctx.circuits, engine.ctx.quota, limiter, _req(), max_attempts=1)
    assert response.xrouter["provider"] == "good1"

    final, final_attempts = await engine._apply_quality_gate(decision, _req(), response, attempts)

    assert final.xrouter["provider"] == "good2"
    assert final.xrouter["quality"]["passed"] is True
    assert final.xrouter["quality_retries"] == 1
    assert [a["status"] for a in final_attempts] == ["success", "success"]


@pytest.mark.asyncio
async def test_llm_grading_final_reasons_and_feedback_when_every_candidate_stays_unsatisfied():
    judge_decision = RoutingDecision(primary=RoutingCandidate(provider_id="judge", model_id="test-model", score=1.0))
    engine, providers, limiter = _engine(llm_grading_enabled=True, max_retries=0, router=_FakeRouter(decision=judge_decision))
    _register(providers, limiter, "good", content="A plausible-looking but actually wrong answer.")
    _register(providers, limiter, "judge", tool_calls=_judgment_tool_call(satisfied=False, feedback="Missing key facts."))
    decision = RoutingDecision(primary=RoutingCandidate(provider_id="good", model_id="test-model", score=1.0))
    response, attempts = await run_chat(decision, providers, engine.ctx.circuits, engine.ctx.quota, limiter, _req(), max_attempts=1)

    final, _ = await engine._apply_quality_gate(decision, _req(), response, attempts)

    assert final.xrouter["quality"]["passed"] is False
    assert "llm_graded_unsatisfactory" in final.xrouter["quality"]["reasons"]
    assert final.xrouter["quality"]["llm_feedback"] == "Missing key facts."


@pytest.mark.asyncio
async def test_llm_grading_falls_open_when_router_has_no_judge_candidate():
    engine, providers, limiter = _engine(llm_grading_enabled=True, router=_FakeRouter(raises=True))
    _register(providers, limiter, "good", content="This is a solid, complete answer.")
    decision = RoutingDecision(primary=RoutingCandidate(provider_id="good", model_id="test-model", score=1.0))
    response, attempts = await run_chat(decision, providers, engine.ctx.circuits, engine.ctx.quota, limiter, _req(), max_attempts=1)

    final, _ = await engine._apply_quality_gate(decision, _req(), response, attempts)

    assert final.xrouter["quality"]["passed"] is True


@pytest.mark.asyncio
async def test_llm_grading_falls_open_when_the_judge_candidate_cannot_actually_be_reached():
    # The router "picks" a judge provider that was never registered in the
    # provider registry -- run_chat itself then raises NoAvailableModelError,
    # a distinct fail-open path from router.select raising.
    judge_decision = RoutingDecision(primary=RoutingCandidate(provider_id="nonexistent-judge", model_id="test-model", score=1.0))
    engine, providers, limiter = _engine(llm_grading_enabled=True, router=_FakeRouter(decision=judge_decision))
    _register(providers, limiter, "good", content="This is a solid, complete answer.")
    decision = RoutingDecision(primary=RoutingCandidate(provider_id="good", model_id="test-model", score=1.0))
    response, attempts = await run_chat(decision, providers, engine.ctx.circuits, engine.ctx.quota, limiter, _req(), max_attempts=1)

    final, _ = await engine._apply_quality_gate(decision, _req(), response, attempts)

    assert final.xrouter["quality"]["passed"] is True


def test_prefer_different_provider_swaps_when_an_alternative_exists():
    engine, _, _ = _engine()
    decision = RoutingDecision(
        primary=RoutingCandidate(provider_id="good", model_id="m1", score=1.0),
        fallback_chain=[
            RoutingCandidate(provider_id="good", model_id="m2", score=0.9),
            RoutingCandidate(provider_id="other", model_id="m3", score=0.5),
        ],
    )
    swapped = engine._prefer_different_provider(decision, "good")
    assert swapped.primary.provider_id == "other"
    assert {c.provider_id for c in swapped.fallback_chain} == {"good", "good"}
    assert len(swapped.fallback_chain) == 2


def test_prefer_different_provider_noop_when_no_alternative_exists():
    engine, _, _ = _engine()
    decision = RoutingDecision(
        primary=RoutingCandidate(provider_id="good", model_id="m1", score=1.0),
        fallback_chain=[RoutingCandidate(provider_id="good", model_id="m2", score=0.9)],
    )
    result = engine._prefer_different_provider(decision, "good")
    assert result is decision


def test_prefer_different_provider_noop_when_primary_is_already_different():
    engine, _, _ = _engine()
    decision = RoutingDecision(primary=RoutingCandidate(provider_id="other", model_id="m1", score=1.0))
    result = engine._prefer_different_provider(decision, "good")
    assert result is decision


def test_prefer_different_provider_noop_when_exclude_is_none():
    engine, _, _ = _engine()
    decision = RoutingDecision(primary=RoutingCandidate(provider_id="good", model_id="m1", score=1.0))
    result = engine._prefer_different_provider(decision, None)
    assert result is decision
