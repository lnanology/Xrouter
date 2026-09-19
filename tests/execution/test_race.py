import asyncio

import pytest

from app.contracts.provider import ProviderConfig
from app.contracts.request import ChatCompletionRequest, ChatMessage
from app.contracts.router import RoutingCandidate, RoutingDecision
from app.core.errors import NoAvailableModelError
from app.core.registry import ProviderRegistry
from app.execution.race import run_race, run_stream_race
from app.observability.events import EventBus
from app.quota.tracker import QuotaTracker
from app.reliability.circuit_breaker import CircuitBreakerRegistry
from app.routing.scheduler import ConcurrencyLimiter
from tests.helpers import FakeProvider, make_model


def _setup(specs: dict[str, dict]):
    """specs: {provider_id: {"behavior": ..., "delay_seconds": ..., "fail_after_chunks": ...}}"""
    providers = ProviderRegistry()
    fakes: dict[str, FakeProvider] = {}
    for pid, spec in specs.items():
        cfg = ProviderConfig(id=pid, name=pid, type="openai_compatible", timeout_seconds=2.0)
        fake = FakeProvider(
            cfg, [make_model(pid)],
            behavior=spec.get("behavior", "success"),
            delay_seconds=spec.get("delay_seconds", 0.0),
            fail_after_chunks=spec.get("fail_after_chunks"),
        )
        providers.register(fake, cfg)
        fakes[pid] = fake
    circuits = CircuitBreakerRegistry()
    quota = QuotaTracker()
    limiter = ConcurrencyLimiter(global_limit=16)
    for pid in specs:
        limiter.configure_provider(pid, 4)
    return providers, circuits, quota, limiter, fakes


def _decision(order: list[str]) -> RoutingDecision:
    candidates = [RoutingCandidate(provider_id=pid, model_id="test-model", score=1.0) for pid in order]
    return RoutingDecision(primary=candidates[0], fallback_chain=candidates[1:])


def _req():
    return ChatCompletionRequest(model="auto", messages=[ChatMessage(role="user", content="hi")], race=True)


def _stream_req():
    return ChatCompletionRequest(
        model="auto", messages=[ChatMessage(role="user", content="hi")], race=True, stream=True,
    )


@pytest.mark.asyncio
async def test_fastest_candidate_wins():
    providers, circuits, quota, limiter, fakes = _setup({
        "slow": {"delay_seconds": 0.2},
        "fast": {"delay_seconds": 0.01},
    })
    response, attempts = await run_race(
        _decision(["slow", "fast"]), providers, circuits, quota, limiter, _req(), max_attempts=3, race_candidate_count=2,
    )
    assert response.xrouter["provider"] == "fast"
    assert response.xrouter["race"] is True
    assert response.xrouter["race_candidates"] == 2
    # give the cancelled loser's task a moment to actually unwind
    await asyncio.sleep(0.3)
    assert fakes["slow"].cancelled is True
    statuses = {a["provider_id"]: a["status"] for a in attempts}
    assert statuses["fast"] == "success"
    assert statuses["slow"] == "cancelled"


@pytest.mark.asyncio
async def test_race_emits_started_and_completed_events():
    providers, circuits, quota, limiter, _ = _setup({"a": {"delay_seconds": 0.0}, "b": {"delay_seconds": 0.05}})
    events = EventBus()
    seen = []
    events.subscribe("race.started", lambda p: seen.append(("started", p)))
    events.subscribe("race.completed", lambda p: seen.append(("completed", p)))
    await run_race(_decision(["a", "b"]), providers, circuits, quota, limiter, _req(), max_attempts=3, race_candidate_count=2, events=events)
    kinds = [k for k, _ in seen]
    assert kinds == ["started", "completed"]
    assert seen[1][1]["winner"] == "a"


@pytest.mark.asyncio
async def test_all_raced_candidates_fail_falls_back_to_rest():
    providers, circuits, quota, limiter, _ = _setup({
        "p1": {"behavior": "server_error"},
        "p2": {"behavior": "timeout"},
        "p3": {"behavior": "success"},
    })
    response, attempts = await run_race(
        _decision(["p1", "p2", "p3"]), providers, circuits, quota, limiter, _req(), max_attempts=3, race_candidate_count=2,
    )
    assert response.xrouter["provider"] == "p3"
    assert response.xrouter["race"] is True
    statuses = {a["provider_id"]: a["status"] for a in attempts}
    assert statuses["p1"] == "failed"
    assert statuses["p2"] == "failed"
    assert statuses["p3"] == "success"


@pytest.mark.asyncio
async def test_all_candidates_including_fallback_fail_raises():
    providers, circuits, quota, limiter, _ = _setup({
        "p1": {"behavior": "server_error"},
        "p2": {"behavior": "timeout"},
    })
    with pytest.raises(NoAvailableModelError) as exc_info:
        await run_race(_decision(["p1", "p2"]), providers, circuits, quota, limiter, _req(), max_attempts=3, race_candidate_count=2)
    assert len(exc_info.value.attempts) == 2
    assert all(a["status"] == "failed" for a in exc_info.value.attempts)


@pytest.mark.asyncio
async def test_race_candidate_count_clamped_to_available_candidates():
    # Only one candidate exists; racing "2 of them" should just run that one.
    providers, circuits, quota, limiter, _ = _setup({"solo": {"delay_seconds": 0.0}})
    response, attempts = await run_race(
        _decision(["solo"]), providers, circuits, quota, limiter, _req(), max_attempts=3, race_candidate_count=2,
    )
    assert response.xrouter["provider"] == "solo"
    assert len(attempts) == 1


@pytest.mark.asyncio
async def test_race_respects_max_attempts_bound():
    # 3 candidates configured but max_attempts=1 should mean only 1 is ever raced.
    providers, circuits, quota, limiter, _ = _setup({
        "p1": {"behavior": "server_error"},
        "p2": {"behavior": "success"},
        "p3": {"behavior": "success"},
    })
    with pytest.raises(NoAvailableModelError) as exc_info:
        await run_race(_decision(["p1", "p2", "p3"]), providers, circuits, quota, limiter, _req(), max_attempts=1, race_candidate_count=2)
    assert len(exc_info.value.attempts) == 1
    assert exc_info.value.attempts[0]["provider_id"] == "p1"


# --- run_stream_race ---------------------------------------------------------

def _attempt_hook():
    """Returns (attempts_log, hook) — hook is passed as on_attempt=..."""
    attempts: list[dict] = []
    return attempts, attempts.append


@pytest.mark.asyncio
async def test_fastest_streaming_candidate_wins_and_streams_in_full():
    providers, circuits, quota, limiter, fakes = _setup({
        "slow": {"delay_seconds": 0.2},
        "fast": {"delay_seconds": 0.01},
    })
    attempts, hook = _attempt_hook()
    chunks = [
        c async for c in run_stream_race(
            _decision(["slow", "fast"]), providers, circuits, quota, limiter, _stream_req(),
            max_attempts=3, race_candidate_count=2, on_attempt=hook,
        )
    ]
    assert len(chunks) == 3
    assert chunks[0].model == "fast/test-model"
    # give the loser's cancelled task a moment to actually unwind
    await asyncio.sleep(0.3)
    assert fakes["slow"].cancelled is True
    statuses = {a["provider_id"]: a["status"] for a in attempts}
    assert statuses["fast"] == "success"
    assert statuses["slow"] == "cancelled"


@pytest.mark.asyncio
async def test_streaming_race_emits_started_and_completed_events():
    providers, circuits, quota, limiter, _ = _setup({"a": {"delay_seconds": 0.0}, "b": {"delay_seconds": 0.05}})
    events = EventBus()
    seen = []
    events.subscribe("race.started", lambda p: seen.append(("started", p)))
    events.subscribe("race.completed", lambda p: seen.append(("completed", p)))
    chunks = [
        c async for c in run_stream_race(
            _decision(["a", "b"]), providers, circuits, quota, limiter, _stream_req(),
            max_attempts=3, race_candidate_count=2, events=events,
        )
    ]
    assert len(chunks) == 3
    kinds = [k for k, _ in seen]
    assert kinds == ["started", "completed"]
    assert seen[1][1]["winner"] == "a"


@pytest.mark.asyncio
async def test_streaming_all_raced_candidates_fail_falls_back_to_rest():
    providers, circuits, quota, limiter, _ = _setup({
        "p1": {"behavior": "server_error"},
        "p2": {"behavior": "timeout"},
        "p3": {"behavior": "success"},
    })
    attempts, hook = _attempt_hook()
    chunks = [
        c async for c in run_stream_race(
            _decision(["p1", "p2", "p3"]), providers, circuits, quota, limiter, _stream_req(),
            max_attempts=3, race_candidate_count=2, on_attempt=hook,
        )
    ]
    assert len(chunks) == 3
    assert chunks[0].model == "p3/test-model"
    statuses = {a["provider_id"]: a["status"] for a in attempts}
    assert statuses["p1"] == "failed"
    assert statuses["p2"] == "failed"
    assert statuses["p3"] == "success"


@pytest.mark.asyncio
async def test_streaming_all_candidates_including_fallback_fail_raises():
    providers, circuits, quota, limiter, _ = _setup({
        "p1": {"behavior": "server_error"},
        "p2": {"behavior": "timeout"},
    })
    with pytest.raises(NoAvailableModelError) as exc_info:
        async for _ in run_stream_race(
            _decision(["p1", "p2"]), providers, circuits, quota, limiter, _stream_req(),
            max_attempts=3, race_candidate_count=2,
        ):
            pass
    assert len(exc_info.value.attempts) == 2
    assert all(a["status"] == "failed" for a in exc_info.value.attempts)


@pytest.mark.asyncio
async def test_streaming_race_skipped_candidates_are_logged():
    # Contrast with run_stream_chat, which deliberately swallows skips --
    # this is new code with no legacy quirk to preserve, and matches
    # run_race's own real behavior of logging every attempt including skips.
    providers, circuits, quota, limiter, _ = _setup({
        "p1": {"behavior": "server_error"},
        "p2": {"behavior": "success"},
    })
    # Trip p1's breaker first (3 consecutive failures at the default threshold).
    for _ in range(3):
        try:
            async for _ in run_stream_race(
                _decision(["p1"]), providers, circuits, quota, limiter, _stream_req(),
                max_attempts=1, race_candidate_count=1,
            ):
                pass
        except NoAvailableModelError:
            pass

    attempts, hook = _attempt_hook()
    chunks = [
        c async for c in run_stream_race(
            _decision(["p1", "p2"]), providers, circuits, quota, limiter, _stream_req(),
            max_attempts=3, race_candidate_count=2, on_attempt=hook,
        )
    ]
    assert len(chunks) == 3
    statuses = {a["provider_id"]: a["status"] for a in attempts}
    assert statuses["p1"] == "skipped"
    assert statuses["p2"] == "success"


@pytest.mark.asyncio
async def test_streaming_race_mid_stream_failure_yields_terminal_error_chunk():
    providers, circuits, quota, limiter, _ = _setup({"p1": {"fail_after_chunks": 1}})
    chunks = [
        c async for c in run_stream_race(
            _decision(["p1"]), providers, circuits, quota, limiter, _stream_req(),
            max_attempts=3, race_candidate_count=1,
        )
    ]
    assert chunks[-1].choices[0].finish_reason == "error"
    assert "interrupted" in chunks[-1].choices[0].delta["content"]


@pytest.mark.asyncio
async def test_streaming_race_closes_a_losing_generator_that_already_got_its_first_chunk():
    # Both candidates are fast enough that their first-chunk attempts
    # routinely land in the same asyncio.wait() completion batch -- proving
    # the "won its own race, lost the outer one" cleanup path actually
    # closes the discarded generator instead of leaking it. Run several
    # trials since exact same-tick completion isn't strictly guaranteed by
    # the event loop; at least one trial landing both in the same batch is
    # enough to prove the cleanup code path works when it's hit.
    async def _one_trial():
        providers, circuits, quota, limiter, fakes = _setup({
            "a": {"delay_seconds": 0.0}, "b": {"delay_seconds": 0.0},
        })
        chunks = [
            c async for c in run_stream_race(
                _decision(["a", "b"]), providers, circuits, quota, limiter, _stream_req(),
                max_attempts=3, race_candidate_count=2,
            )
        ]
        assert len(chunks) == 3
        winner = chunks[0].model.split("/")[0]
        loser = "b" if winner == "a" else "a"
        return fakes[loser].stream_aborted

    results = [await _one_trial() for _ in range(20)]
    assert any(results), "expected at least one trial to close a same-batch losing generator"
