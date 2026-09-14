import pytest

from app.contracts.provider import ProviderConfig
from app.contracts.request import ChatCompletionRequest, ChatMessage
from app.contracts.router import RoutingCandidate, RoutingDecision
from app.core.errors import NoAvailableModelError
from app.core.registry import ProviderRegistry
from app.quota.tracker import QuotaTracker
from app.reliability.circuit_breaker import CircuitBreakerRegistry, CircuitState
from app.routing.fallback import run_chat, run_stream_chat
from app.routing.scheduler import ConcurrencyLimiter
from tests.helpers import FakeProvider, make_model


def _setup(behaviors: dict[str, str], fail_after_chunks: dict[str, int] | None = None):
    providers = ProviderRegistry()
    for pid, behavior in behaviors.items():
        cfg = ProviderConfig(id=pid, name=pid, type="openai_compatible", timeout_seconds=2.0)
        fa = (fail_after_chunks or {}).get(pid)
        fake = FakeProvider(cfg, [make_model(pid)], behavior=behavior, fail_after_chunks=fa)
        providers.register(fake, cfg)
    circuits = CircuitBreakerRegistry()
    quota = QuotaTracker()
    limiter = ConcurrencyLimiter(global_limit=16)
    for pid in behaviors:
        limiter.configure_provider(pid, 4)
    return providers, circuits, quota, limiter


def _decision(order: list[str]) -> RoutingDecision:
    candidates = [RoutingCandidate(provider_id=pid, model_id="test-model", score=1.0) for pid in order]
    return RoutingDecision(primary=candidates[0], fallback_chain=candidates[1:])


def _req(stream=False):
    return ChatCompletionRequest(model="auto", messages=[ChatMessage(role="user", content="hi")], stream=stream)


@pytest.mark.asyncio
async def test_success_on_primary():
    providers, circuits, quota, limiter = _setup({"p1": "success"})
    response, attempts = await run_chat(_decision(["p1"]), providers, circuits, quota, limiter, _req(), max_attempts=3)
    assert response.xrouter["provider"] == "p1"
    assert len(attempts) == 1
    assert attempts[0]["status"] == "success"


@pytest.mark.asyncio
async def test_timeout_falls_back_to_next_provider():
    providers, circuits, quota, limiter = _setup({"p1": "timeout", "p2": "success"})
    response, attempts = await run_chat(_decision(["p1", "p2"]), providers, circuits, quota, limiter, _req(), max_attempts=3)
    assert response.xrouter["provider"] == "p2"
    assert attempts[0]["status"] == "failed"
    assert attempts[1]["status"] == "success"


@pytest.mark.asyncio
async def test_rate_limit_falls_back_and_records_quota():
    providers, circuits, quota, limiter = _setup({"p1": "rate_limit", "p2": "success"})
    response, attempts = await run_chat(_decision(["p1", "p2"]), providers, circuits, quota, limiter, _req(), max_attempts=3)
    assert response.xrouter["provider"] == "p2"
    snap = quota.snapshot()
    assert snap["p1"]["recent_429s"] == 1


@pytest.mark.asyncio
async def test_server_error_falls_back():
    providers, circuits, quota, limiter = _setup({"p1": "server_error", "p2": "success"})
    response, _ = await run_chat(_decision(["p1", "p2"]), providers, circuits, quota, limiter, _req(), max_attempts=3)
    assert response.xrouter["provider"] == "p2"


@pytest.mark.asyncio
async def test_all_providers_down_raises_structured_error():
    providers, circuits, quota, limiter = _setup({"p1": "server_error", "p2": "timeout"})
    with pytest.raises(NoAvailableModelError) as exc_info:
        await run_chat(_decision(["p1", "p2"]), providers, circuits, quota, limiter, _req(), max_attempts=3)
    assert len(exc_info.value.attempts) == 2
    assert all(a["status"] == "failed" for a in exc_info.value.attempts)


@pytest.mark.asyncio
async def test_repeated_failures_open_circuit_breaker():
    providers, circuits, quota, limiter = _setup({"p1": "server_error", "p2": "success"})
    # 3 consecutive requests to p1 should trip its breaker (default threshold=3).
    for _ in range(3):
        try:
            await run_chat(_decision(["p1"]), providers, circuits, quota, limiter, _req(), max_attempts=1)
        except NoAvailableModelError:
            pass
    assert circuits.get("p1").state == CircuitState.OPEN
    # A subsequent request preferring p1 but chained to p2 should skip straight to p2.
    response, attempts = await run_chat(_decision(["p1", "p2"]), providers, circuits, quota, limiter, _req(), max_attempts=3)
    assert response.xrouter["provider"] == "p2"
    assert attempts[0]["status"] == "skipped"


@pytest.mark.asyncio
async def test_streaming_success_yields_all_chunks():
    providers, circuits, quota, limiter = _setup({"p1": "success"})
    chunks = [c async for c in run_stream_chat(_decision(["p1"]), providers, circuits, quota, limiter, _req(True), max_attempts=3)]
    assert len(chunks) == 3
    assert chunks[0].choices[0].delta["content"] == "chunk0"


@pytest.mark.asyncio
async def test_streaming_falls_back_when_first_chunk_fails():
    providers, circuits, quota, limiter = _setup({"p1": "timeout", "p2": "success"})
    chunks = [c async for c in run_stream_chat(_decision(["p1", "p2"]), providers, circuits, quota, limiter, _req(True), max_attempts=3)]
    assert len(chunks) == 3
    assert chunks[0].model == "p2/test-model"


@pytest.mark.asyncio
async def test_streaming_mid_stream_failure_yields_error_marker_not_silent_fallback():
    providers, circuits, quota, limiter = _setup({"p1": "success"}, fail_after_chunks={"p1": 1})
    chunks = [c async for c in run_stream_chat(_decision(["p1"]), providers, circuits, quota, limiter, _req(True), max_attempts=3)]
    assert chunks[-1].choices[0].finish_reason == "error"
    assert "interrupted" in chunks[-1].choices[0].delta["content"]
