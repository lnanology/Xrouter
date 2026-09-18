"""Unit tests for SelfHealer (Phase 5's fifth and last piece) -- the
confirm-cycle streak mechanism (disable/recover), the forget() admin
handback, snapshot() shape, and start()/stop() lifecycle (mirroring
tests/reliability/test_benchmark.py's own tests for that shape).

Simpler than every other Phase 5 piece: no Database/MetricsRepository at
all, just a real ProviderRegistry + CircuitBreakerRegistry + EventBus --
this piece never touches persistence or the network. Scenarios are
driven directly via providers.set_health()/circuits.get(pid).
record_failure()/record_success()/force_open(), the real registry/breaker
API, never provider.health() (SelfHealer never re-probes)."""
from __future__ import annotations

import asyncio

import pytest

from app.contracts.provider import ProviderConfig, ProviderHealth, ProviderStatus
from app.observability.events import EventBus
from app.reliability.circuit_breaker import CircuitBreakerRegistry
from app.reliability.self_healer import SelfHealer
from app.core.registry import ProviderRegistry
from tests.helpers import FakeProvider, make_model


async def _build(
    provider_ids: list[str], enabled: bool = True, confirm_cycles: int = 3, interval_seconds: float = 30.0,
) -> tuple[SelfHealer, ProviderRegistry, CircuitBreakerRegistry, EventBus]:
    providers = ProviderRegistry()
    for pid in provider_ids:
        cfg = ProviderConfig(id=pid, name=pid, type="openai_compatible")
        providers.register(FakeProvider(cfg, [make_model(pid)]), cfg, enabled=True)
    circuits = CircuitBreakerRegistry()
    events = EventBus()
    healer = SelfHealer(providers, circuits, events, enabled=enabled, interval_seconds=interval_seconds, confirm_cycles=confirm_cycles)
    return healer, providers, circuits, events


def _mark_unhealthy(providers: ProviderRegistry, pid: str) -> None:
    providers.set_health(pid, ProviderHealth(status=ProviderStatus.OFFLINE))


def _mark_healthy(providers: ProviderRegistry, pid: str) -> None:
    providers.set_health(pid, ProviderHealth(status=ProviderStatus.HEALTHY))


# --- healthy providers are left alone ----------------------------------

@pytest.mark.asyncio
async def test_a_healthy_provider_stays_enabled_and_untouched():
    healer, providers, _, _ = await _build(["p1"])
    for _ in range(5):
        result = await healer.run_once()
        assert result["actions"] == {}
    assert providers.is_enabled("p1") is True


# --- disable path --------------------------------------------------------

@pytest.mark.asyncio
async def test_staying_unhealthy_below_confirm_cycles_leaves_it_enabled():
    healer, providers, _, _ = await _build(["p1"], confirm_cycles=3)
    _mark_unhealthy(providers, "p1")

    for _ in range(2):  # confirm_cycles - 1
        result = await healer.run_once()
        assert result["actions"] == {}
    assert providers.is_enabled("p1") is True


@pytest.mark.asyncio
async def test_reaching_confirm_cycles_of_bad_health_disables_the_provider():
    healer, providers, _, events = await _build(["p1"], confirm_cycles=3)
    _mark_unhealthy(providers, "p1")
    seen = []
    events.subscribe("self_healing.disabled", lambda p: seen.append(p))

    result = None
    for _ in range(3):
        result = await healer.run_once()

    assert result["actions"] == {"p1": "disabled"}
    assert providers.is_enabled("p1") is False
    assert seen == [{"provider_id": "p1"}]
    assert "p1" in healer.snapshot()["disabled_by_self_healing"]


@pytest.mark.asyncio
async def test_a_single_healthy_tick_resets_an_in_progress_bad_streak():
    healer, providers, _, _ = await _build(["p1"], confirm_cycles=3)
    _mark_unhealthy(providers, "p1")
    await healer.run_once()
    await healer.run_once()  # 2 bad ticks -- one short of disabling

    _mark_healthy(providers, "p1")
    await healer.run_once()  # resets the streak

    _mark_unhealthy(providers, "p1")
    await healer.run_once()
    result = await healer.run_once()  # only 2 bad ticks since the reset

    assert result["actions"] == {}
    assert providers.is_enabled("p1") is True


@pytest.mark.asyncio
async def test_an_open_circuit_alone_also_triggers_a_disable():
    healer, providers, circuits, _ = await _build(["p1"], confirm_cycles=3)
    # Health is left HEALTHY throughout -- only the breaker signal is bad.
    breaker = circuits.get("p1")
    breaker.force_open(cooldown_seconds=99999.0)  # stays OPEN for the whole test

    result = None
    for _ in range(3):
        result = await healer.run_once()

    assert result["actions"] == {"p1": "disabled"}
    assert providers.is_enabled("p1") is False


# --- recovery path ---------------------------------------------------------

@pytest.mark.asyncio
async def test_a_self_disabled_provider_recovers_after_confirm_cycles_of_good_health():
    healer, providers, _, events = await _build(["p1"], confirm_cycles=2)
    _mark_unhealthy(providers, "p1")
    seen = []
    events.subscribe("self_healing.recovered", lambda p: seen.append(p))

    await healer.run_once()
    await healer.run_once()  # disabled here
    assert providers.is_enabled("p1") is False

    _mark_healthy(providers, "p1")
    await healer.run_once()
    result = await healer.run_once()

    assert result["actions"] == {"p1": "recovered"}
    assert providers.is_enabled("p1") is True
    assert seen == [{"provider_id": "p1"}]
    assert healer.snapshot()["disabled_by_self_healing"] == {}


@pytest.mark.asyncio
async def test_a_provider_never_managed_by_self_healing_is_never_a_recovery_candidate():
    healer, providers, _, _ = await _build(["p1"], confirm_cycles=2)
    providers.set_enabled("p1", False)  # disabled by something other than SelfHealer
    _mark_healthy(providers, "p1")

    for _ in range(5):
        result = await healer.run_once()
        assert result["actions"] == {}
    assert providers.is_enabled("p1") is False  # never touched


# --- forget() --------------------------------------------------------------

@pytest.mark.asyncio
async def test_forget_after_self_disable_makes_it_stick_disabled():
    healer, providers, _, _ = await _build(["p1"], confirm_cycles=2)
    _mark_unhealthy(providers, "p1")
    await healer.run_once()
    await healer.run_once()
    assert providers.is_enabled("p1") is False

    healer.forget("p1")  # simulates an admin's explicit disable of it

    _mark_healthy(providers, "p1")
    for _ in range(5):
        result = await healer.run_once()
        assert result["actions"] == {}
    assert providers.is_enabled("p1") is False  # stays off -- no longer a recovery candidate


@pytest.mark.asyncio
async def test_forget_after_manual_reenable_requires_a_full_fresh_streak():
    healer, providers, _, _ = await _build(["p1"], confirm_cycles=3)
    _mark_unhealthy(providers, "p1")
    await healer.run_once()
    await healer.run_once()  # 2 bad ticks accumulated, one short of disabling

    providers.set_enabled("p1", True)  # simulates an admin manually re-enabling it
    healer.forget("p1")  # clears the stale streak

    # Still unhealthy -- must take a full fresh confirm_cycles streak, not
    # just 1 more tick (which is all that would be left without forget()).
    result = await healer.run_once()
    assert result["actions"] == {}
    assert providers.is_enabled("p1") is True

    result = await healer.run_once()
    assert result["actions"] == {}
    assert providers.is_enabled("p1") is True

    result = await healer.run_once()
    assert result["actions"] == {"p1": "disabled"}


# --- snapshot() --------------------------------------------------------------

@pytest.mark.asyncio
async def test_snapshot_shape():
    healer, providers, _, _ = await _build(["p1"], enabled=True, confirm_cycles=3, interval_seconds=45.0)
    assert healer.snapshot() == {
        "enabled": True, "interval_seconds": 45.0, "confirm_cycles": 3, "disabled_by_self_healing": {},
    }

    _mark_unhealthy(providers, "p1")
    for _ in range(3):
        await healer.run_once()

    snap = healer.snapshot()
    assert snap["enabled"] is True
    assert set(snap["disabled_by_self_healing"]) == {"p1"}
    assert isinstance(snap["disabled_by_self_healing"]["p1"], float)


# --- start()/stop() lifecycle (mirrors BenchmarkScheduler's own tests) -----

@pytest.mark.asyncio
async def test_stop_before_start_is_a_safe_noop():
    healer, _, _, _ = await _build(["p1"], interval_seconds=0.05)
    await healer.stop()  # must not raise


@pytest.mark.asyncio
async def test_start_runs_in_the_background_and_stop_tears_it_down_cleanly():
    healer, providers, _, _ = await _build(["p1"], confirm_cycles=1, interval_seconds=0.05)
    _mark_unhealthy(providers, "p1")

    healer.start()
    assert healer._task is not None
    await asyncio.sleep(0.2)  # give the loop at least one full cycle to run
    await healer.stop()
    assert healer._task is None

    assert providers.is_enabled("p1") is False  # at least one cycle ran and disabled it


@pytest.mark.asyncio
async def test_start_is_idempotent():
    healer, _, _, _ = await _build(["p1"], interval_seconds=1.0)
    healer.start()
    task = healer._task
    healer.start()  # second call must not replace the running task
    assert healer._task is task
    await healer.stop()
