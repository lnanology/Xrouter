import asyncio

import pytest

from app.contracts.provider import ProviderConfig
from app.core.registry import ModelRegistry, ProviderRegistry
from app.reliability.benchmark import BenchmarkScheduler
from app.storage.database import Database
from app.storage.repositories.metrics import MetricsRepository
from tests.helpers import FakeProvider, make_model


async def _build(tmp_path, provider_specs: dict) -> tuple[BenchmarkScheduler, MetricsRepository]:
    """provider_specs: {provider_id: {"behavior": ..., "enabled": bool, "no_model": bool}}"""
    db = Database(str(tmp_path / "test.sqlite3"))
    await db.init()
    metrics_repo = MetricsRepository(db)

    providers = ProviderRegistry()
    models = ModelRegistry()
    for pid, spec in provider_specs.items():
        spec = dict(spec)
        enabled = spec.pop("enabled", True)
        no_model = spec.pop("no_model", False)
        cfg = ProviderConfig(id=pid, name=pid, type="openai_compatible", timeout_seconds=2.0)
        fake = FakeProvider(cfg, [] if no_model else [make_model(pid)], **spec)
        providers.register(fake, cfg, enabled=enabled)
    await models.refresh(providers)

    scheduler = BenchmarkScheduler(providers, models, metrics_repo, interval_seconds=0.05)
    return scheduler, metrics_repo


# --- run_once ------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_once_returns_and_persists_one_result_per_enabled_provider(tmp_path):
    scheduler, metrics_repo = await _build(tmp_path, {"p1": {}, "p2": {}})
    results = await scheduler.run_once()

    assert {r["provider"] for r in results} == {"p1", "p2"}
    assert all(r["success"] for r in results)
    assert all(r["latency_ms"] >= 0 for r in results)

    persisted = await metrics_repo.recent_benchmarks(limit=10)
    assert {row["provider_id"] for row in persisted} == {"p1", "p2"}


@pytest.mark.asyncio
async def test_run_once_records_a_failing_provider_without_aborting_the_batch(tmp_path):
    scheduler, metrics_repo = await _build(
        tmp_path, {"broken": {"behavior": "server_error"}, "fine": {}},
    )
    results = await scheduler.run_once()

    by_provider = {r["provider"]: r for r in results}
    assert by_provider["broken"]["success"] is False
    assert by_provider["fine"]["success"] is True

    persisted = await metrics_repo.recent_benchmarks(limit=10)
    persisted_by_provider = {row["provider_id"]: row for row in persisted}
    assert persisted_by_provider["broken"]["success"] is False
    assert persisted_by_provider["fine"]["success"] is True


@pytest.mark.asyncio
async def test_run_once_skips_disabled_providers(tmp_path):
    scheduler, _ = await _build(tmp_path, {"on": {}, "off": {"enabled": False}})
    results = await scheduler.run_once()
    assert {r["provider"] for r in results} == {"on"}


@pytest.mark.asyncio
async def test_run_once_skips_providers_with_no_models(tmp_path):
    scheduler, _ = await _build(tmp_path, {"has_model": {}, "modelless": {"no_model": True}})
    results = await scheduler.run_once()
    assert {r["provider"] for r in results} == {"has_model"}


@pytest.mark.asyncio
async def test_run_once_with_no_providers_returns_empty_list(tmp_path):
    scheduler, _ = await _build(tmp_path, {})
    assert await scheduler.run_once() == []


# --- start()/stop() lifecycle (same shape as HealthMonitor) --------------

@pytest.mark.asyncio
async def test_stop_before_start_is_a_safe_noop(tmp_path):
    scheduler, _ = await _build(tmp_path, {"p1": {}})
    await scheduler.stop()  # must not raise


@pytest.mark.asyncio
async def test_start_runs_in_the_background_and_stop_tears_it_down_cleanly(tmp_path):
    scheduler, metrics_repo = await _build(tmp_path, {"p1": {}})
    scheduler.start()
    assert scheduler._task is not None
    # Interval is 0.05s -- give the loop at least one full cycle to run.
    await asyncio.sleep(0.15)
    await scheduler.stop()
    assert scheduler._task is None

    persisted = await metrics_repo.recent_benchmarks(limit=10)
    assert len(persisted) >= 1


@pytest.mark.asyncio
async def test_start_is_idempotent(tmp_path):
    scheduler, _ = await _build(tmp_path, {"p1": {}})
    scheduler.start()
    task = scheduler._task
    scheduler.start()  # second call must not replace the running task
    assert scheduler._task is task
    await scheduler.stop()
