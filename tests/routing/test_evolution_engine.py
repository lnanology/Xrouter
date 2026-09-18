"""Unit tests for EvolutionEngine (Phase 5's fourth piece) -- the pending-
nudge lifecycle (record -> evaluate -> confirm/revert), the exclude
threading into PolicyLearner.run_once(), snapshot() shape, and the
start()/_loop()/stop() background-task shape (mirroring tests/reliability/
test_benchmark.py's own tests for that exact shape).

Built against a real Database + MetricsRepository + PolicyLearner, same
pattern tests/routing/test_policy_learner.py already uses -- no mocking of
the collaborators this engine actually depends on."""
from __future__ import annotations

import asyncio
import time
from dataclasses import fields

import pytest

from app.contracts.policy import PolicyWeights
from app.routing.evolution_engine import EvolutionEngine
from app.routing.policy_learner import PolicyLearner
from app.routing.router import POLICY_WEIGHTS
from app.storage.database import Database
from app.storage.repositories.metrics import MetricsRepository


async def _build(
    tmp_path, variants: list[str], enabled: bool = True, min_samples: int = 5, min_margin: float = 0.05,
    **evo_kwargs,
) -> tuple[EvolutionEngine, PolicyLearner, MetricsRepository]:
    db = Database(str(tmp_path / "test.sqlite3"))
    await db.init()
    repo = MetricsRepository(db)
    learner = PolicyLearner(variants, repo, enabled=True, min_samples=min_samples, min_margin=min_margin)
    evo_kwargs.setdefault("evaluation_samples", 5)
    evo_kwargs.setdefault("rollback_tolerance", 0.02)
    engine = EvolutionEngine(learner, repo, enabled=enabled, **evo_kwargs)
    return engine, learner, repo


async def _seed(repo: MetricsRepository, variant: str, successes: int, failures: int, latency_ms: float = 0.0) -> None:
    for _ in range(successes):
        await repo.record_ab_result(variant, success=True, latency_ms=latency_ms, quality_score=None)
    for _ in range(failures):
        await repo.record_ab_result(variant, success=False, latency_ms=latency_ms, quality_score=None)


# --- run_once() records a fresh nudge into pending --------------------------

@pytest.mark.asyncio
async def test_run_once_records_a_new_nudge_into_pending(tmp_path):
    engine, _, repo = await _build(tmp_path, ["quality", "balanced"])
    await _seed(repo, "quality", successes=5, failures=0)
    await _seed(repo, "balanced", successes=0, failures=5)

    result = await engine.run_once()

    assert result["evaluated"] == {}
    assert result["learn"]["applied"] is True
    assert result["pending"] == ["balanced"]
    assert "balanced" in engine._pending
    pending = engine._pending["balanced"]
    assert pending["previous_weights"] is None  # first-ever nudge -- no prior override existed
    assert pending["baseline_fitness"] == pytest.approx(result["learn"]["adjustments"]["balanced"]["fitness_before"])
    assert isinstance(pending["applied_at"], float)


# --- pending stays pending without enough fresh evidence ---------------------

@pytest.mark.asyncio
async def test_pending_stays_pending_without_enough_fresh_samples(tmp_path):
    engine, _, repo = await _build(tmp_path, ["quality", "balanced"])
    await _seed(repo, "quality", successes=5, failures=0)
    await _seed(repo, "balanced", successes=0, failures=5)
    await engine.run_once()
    assert "balanced" in engine._pending

    # No new rows recorded after the nudge -- nothing fresh to evaluate yet.
    second = await engine.run_once()
    assert second["evaluated"] == {}
    assert "balanced" in engine._pending
    # A policy still awaiting evaluation must not be nudged again.
    assert second["learn"]["adjustments"] == {}

    third = await engine.run_once()
    assert third["evaluated"] == {}
    assert "balanced" in engine._pending


# --- _evaluate_pending(): confirm / revert -----------------------------------

@pytest.mark.asyncio
async def test_evaluate_pending_confirms_when_fresh_fitness_holds_up(tmp_path):
    engine, learner, repo = await _build(tmp_path, ["quality", "balanced"])
    engine._pending["balanced"] = {"applied_at": time.time(), "previous_weights": None, "baseline_fitness": 0.0}
    await asyncio.sleep(0.01)
    # Fresh evidence matches the same (bad) baseline -- not a further drop.
    await _seed(repo, "balanced", successes=0, failures=5)

    evaluated = await engine._evaluate_pending()

    assert evaluated["balanced"]["outcome"] == "confirmed"
    assert "balanced" not in engine._pending
    assert learner._learned == {}  # revert() never invoked -- nothing was touched


@pytest.mark.asyncio
async def test_evaluate_pending_reverts_when_fresh_fitness_drops_below_tolerance(tmp_path):
    engine, learner, repo = await _build(tmp_path, ["quality", "balanced"])
    previous = {f.name: getattr(POLICY_WEIGHTS["balanced"], f.name) for f in fields(PolicyWeights)}
    learner._learned["balanced"] = PolicyWeights(**{**previous, "quality": 3.0})  # simulates a prior nudge's result
    engine._pending["balanced"] = {
        "applied_at": time.time(), "previous_weights": previous, "baseline_fitness": 0.5,
    }
    await asyncio.sleep(0.01)
    # Fresh evidence: fitness ~0.0, well below baseline(0.5) - tolerance(0.02).
    await _seed(repo, "balanced", successes=0, failures=5)

    evaluated = await engine._evaluate_pending()

    assert evaluated["balanced"]["outcome"] == "reverted"
    assert "balanced" not in engine._pending
    restored = learner._learned["balanced"]
    for f in fields(PolicyWeights):
        assert getattr(restored, f.name) == pytest.approx(previous[f.name])


@pytest.mark.asyncio
async def test_evaluate_pending_revert_with_none_previous_weights_removes_the_override(tmp_path):
    engine, learner, repo = await _build(tmp_path, ["quality", "balanced"])
    learner._learned["balanced"] = PolicyWeights(quality=3.0)  # simulates the (only ever) prior nudge
    engine._pending["balanced"] = {"applied_at": time.time(), "previous_weights": None, "baseline_fitness": 0.5}
    await asyncio.sleep(0.01)
    await _seed(repo, "balanced", successes=0, failures=5)

    evaluated = await engine._evaluate_pending()

    assert evaluated["balanced"]["outcome"] == "reverted"
    assert "balanced" not in learner._learned  # None -> override fully removed, falls back to base


# --- exclude threading --------------------------------------------------------

@pytest.mark.asyncio
async def test_exclude_blocks_every_policy_still_awaiting_evaluation(tmp_path):
    engine, _, repo = await _build(tmp_path, ["quality", "fastest", "balanced"])
    await _seed(repo, "quality", successes=5, failures=0)
    await _seed(repo, "fastest", successes=0, failures=5)
    await _seed(repo, "balanced", successes=0, failures=5)

    first = await engine.run_once()
    assert set(engine._pending) == {"fastest", "balanced"}
    assert first["learn"]["applied"] is True

    # Both losers are now pending evaluation -- neither should be re-nudged
    # before it's judged, even though nothing fresh has been evaluated yet.
    second = await engine.run_once()
    assert second["learn"]["adjustments"] == {}
    assert set(engine._pending) == {"fastest", "balanced"}


# --- snapshot() ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_snapshot_reports_enabled_interval_and_pending(tmp_path):
    engine, _, repo = await _build(tmp_path, ["quality", "balanced"], enabled=True, interval_seconds=120.0)
    assert engine.snapshot() == {"enabled": True, "interval_seconds": 120.0, "pending": {}}

    await _seed(repo, "quality", successes=5, failures=0)
    await _seed(repo, "balanced", successes=0, failures=5)
    await engine.run_once()

    snap = engine.snapshot()
    assert snap["enabled"] is True
    assert snap["interval_seconds"] == 120.0
    assert set(snap["pending"]) == {"balanced"}


# --- start()/stop() lifecycle (mirrors BenchmarkScheduler's own tests) -------

@pytest.mark.asyncio
async def test_stop_before_start_is_a_safe_noop(tmp_path):
    engine, _, _ = await _build(tmp_path, ["quality", "balanced"], interval_seconds=0.05)
    await engine.stop()  # must not raise


@pytest.mark.asyncio
async def test_start_runs_in_the_background_and_stop_tears_it_down_cleanly(tmp_path):
    engine, _, repo = await _build(tmp_path, ["quality", "balanced"], interval_seconds=0.05)
    await _seed(repo, "quality", successes=5, failures=0)
    await _seed(repo, "balanced", successes=0, failures=5)

    engine.start()
    assert engine._task is not None
    await asyncio.sleep(0.2)  # give the loop at least one full cycle to run
    await engine.stop()
    assert engine._task is None

    assert "balanced" in engine._pending  # at least one cycle ran and recorded a nudge


@pytest.mark.asyncio
async def test_start_is_idempotent(tmp_path):
    engine, _, _ = await _build(tmp_path, ["quality", "balanced"], interval_seconds=1.0)
    engine.start()
    task = engine._task
    engine.start()  # second call must not replace the running task
    assert engine._task is task
    await engine.stop()
