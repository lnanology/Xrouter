"""Unit tests for PolicyLearner (Phase 5's third piece) -- fitness/nudge
math and the weights_for()/run_once()/snapshot() surface, against a real
MetricsRepository (rows inserted via record_ab_result, same pattern
tests/routing/test_ab_router.py already uses for ABRouter). Engine/router
wiring is covered separately in tests/routing/test_router.py."""
from __future__ import annotations

from dataclasses import fields

import pytest

from app.contracts.policy import PolicyWeights
from app.routing.policy_learner import MAX_WEIGHT, MIN_WEIGHT, PolicyLearner, _clamp, _nudge
from app.routing.router import POLICY_WEIGHTS
from app.storage.database import Database
from app.storage.repositories.metrics import MetricsRepository


async def _build(tmp_path, variants: list[str], enabled: bool = True, **kwargs) -> tuple[PolicyLearner, MetricsRepository]:
    db = Database(str(tmp_path / "test.sqlite3"))
    await db.init()
    metrics_repo = MetricsRepository(db)
    learner = PolicyLearner(variants, metrics_repo, enabled=enabled, **kwargs)
    return learner, metrics_repo


async def _seed(repo: MetricsRepository, variant: str, successes: int, failures: int, latency_ms: float = 0.0) -> None:
    for _ in range(successes):
        await repo.record_ab_result(variant, success=True, latency_ms=latency_ms, quality_score=None)
    for _ in range(failures):
        await repo.record_ab_result(variant, success=False, latency_ms=latency_ms, quality_score=None)


# --- _nudge / _clamp (pure helpers) -------------------------------------

def test_nudge_moves_current_toward_target_by_learning_rate():
    current = PolicyWeights(quality=1.0, speed=1.0, reliability=1.0, cost=1.0, quota_risk=1.0, local_preference=1.0)
    target = PolicyWeights(quality=2.5, speed=0.5, reliability=1.2, cost=0.3, quota_risk=0.8, local_preference=0.6)

    result = _nudge(current, target, learning_rate=0.15)

    assert result.quality == pytest.approx(1.0 + 0.15 * (2.5 - 1.0))
    assert result.speed == pytest.approx(1.0 + 0.15 * (0.5 - 1.0))
    assert result.reliability == pytest.approx(1.0 + 0.15 * (1.2 - 1.0))
    assert result.cost == pytest.approx(1.0 + 0.15 * (0.3 - 1.0))
    assert result.quota_risk == pytest.approx(1.0 + 0.15 * (0.8 - 1.0))
    assert result.local_preference == pytest.approx(1.0 + 0.15 * (0.6 - 1.0))


def test_nudge_clamps_to_safety_bounds():
    over = _nudge(PolicyWeights(quality=4.9), PolicyWeights(quality=10.0), learning_rate=1.0)
    assert over.quality == pytest.approx(MAX_WEIGHT)

    under = _nudge(PolicyWeights(quality=0.1), PolicyWeights(quality=-5.0), learning_rate=1.0)
    assert under.quality == pytest.approx(MIN_WEIGHT)


def test_clamp_is_a_noop_within_bounds():
    assert _clamp(1.23) == pytest.approx(1.23)


# --- weights_for() -------------------------------------------------------

@pytest.mark.asyncio
async def test_weights_for_returns_base_when_disabled_even_with_a_stored_override(tmp_path):
    learner, repo = await _build(tmp_path, ["quality", "balanced"], enabled=False)
    await _seed(repo, "quality", successes=20, failures=0)
    await _seed(repo, "balanced", successes=0, failures=20)
    await learner.run_once()  # computes + stores an override even while disabled (a safe preview)

    base = PolicyWeights(quality=1.0)
    assert learner.weights_for("balanced", base) is base


@pytest.mark.asyncio
async def test_weights_for_returns_base_when_no_override_learned_yet(tmp_path):
    learner, _ = await _build(tmp_path, ["quality", "balanced"], enabled=True)
    base = PolicyWeights(quality=1.0)
    assert learner.weights_for("quality", base) is base


# --- run_once() ------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_once_reports_insufficient_samples_below_min_samples(tmp_path):
    learner, repo = await _build(tmp_path, ["quality", "balanced"], enabled=True, min_samples=20)
    await _seed(repo, "quality", successes=5, failures=0)  # below min_samples
    await _seed(repo, "balanced", successes=20, failures=0)

    result = await learner.run_once()

    assert result == {
        "applied": False, "reason": "insufficient_samples",
        "eligible_variants": ["balanced"], "min_samples": 20,
    }
    assert learner._learned == {}


@pytest.mark.asyncio
async def test_run_once_nudges_the_loser_toward_the_winner_by_exact_amount(tmp_path):
    learner, repo = await _build(tmp_path, ["quality", "balanced"], enabled=True)
    await _seed(repo, "quality", successes=20, failures=0)   # success_rate 1.0
    await _seed(repo, "balanced", successes=0, failures=20)  # success_rate 0.0

    result = await learner.run_once()

    assert result["applied"] is True
    assert set(result["adjustments"]) == {"balanced"}
    assert result["adjustments"]["balanced"]["toward"] == "quality"

    quality_base = POLICY_WEIGHTS["quality"]
    balanced_base = POLICY_WEIGHTS["balanced"]
    learned = learner._learned["balanced"]
    for f in fields(PolicyWeights):
        expected = getattr(balanced_base, f.name) + 0.15 * (getattr(quality_base, f.name) - getattr(balanced_base, f.name))
        assert getattr(learned, f.name) == pytest.approx(expected)
        assert result["adjustments"]["balanced"]["weights"][f.name] == pytest.approx(expected)


@pytest.mark.asyncio
async def test_run_once_skips_a_loser_within_the_margin_threshold(tmp_path):
    learner, repo = await _build(tmp_path, ["quality", "balanced"], enabled=True, min_margin=0.05)
    # Exact tie (0.5 vs 0.5 success rate, zero latency/quality contribution)
    # -- a 0.0 fitness gap, well under the 0.05 margin threshold.
    await _seed(repo, "quality", successes=10, failures=10)
    await _seed(repo, "balanced", successes=10, failures=10)

    result = await learner.run_once()

    assert result["applied"] is False
    assert result["adjustments"] == {}
    assert learner._learned == {}


@pytest.mark.asyncio
async def test_run_once_with_three_variants_only_nudges_the_pairing_that_clears_margin(tmp_path):
    learner, repo = await _build(tmp_path, ["quality", "fastest", "balanced"], enabled=True, min_samples=20)
    await _seed(repo, "quality", successes=25, failures=0)    # success_rate 1.00 -- winner
    await _seed(repo, "fastest", successes=24, failures=1)    # success_rate 0.96 -- margin 0.04, below 0.05 threshold
    await _seed(repo, "balanced", successes=5, failures=20)   # success_rate 0.20 -- margin 0.80, clears threshold

    result = await learner.run_once()

    assert result["applied"] is True
    assert set(result["adjustments"]) == {"balanced"}
    assert "fastest" not in learner._learned
    assert "balanced" in learner._learned


@pytest.mark.asyncio
async def test_run_once_compounds_across_repeated_calls(tmp_path):
    learner, repo = await _build(tmp_path, ["quality", "balanced"], enabled=True)
    await _seed(repo, "quality", successes=20, failures=0)
    await _seed(repo, "balanced", successes=0, failures=20)

    first = await learner.run_once()
    assert first["applied"] is True
    first_balanced = learner._learned["balanced"]

    second = await learner.run_once()
    assert second["applied"] is True
    second_balanced = learner._learned["balanced"]

    quality_base = POLICY_WEIGHTS["quality"]
    for f in fields(PolicyWeights):
        expected = _clamp(
            getattr(first_balanced, f.name) + 0.15 * (getattr(quality_base, f.name) - getattr(first_balanced, f.name))
        )
        assert getattr(second_balanced, f.name) == pytest.approx(expected)
        # A genuine second step, not a no-op repeat of the first.
        assert getattr(second_balanced, f.name) != pytest.approx(getattr(first_balanced, f.name))


# --- snapshot() ------------------------------------------------------------

@pytest.mark.asyncio
async def test_snapshot_reports_enabled_variants_and_learned_overrides(tmp_path):
    learner, repo = await _build(tmp_path, ["quality", "balanced"], enabled=True)
    assert learner.snapshot() == {"enabled": True, "variants": ["quality", "balanced"], "learned_overrides": {}}

    await _seed(repo, "quality", successes=20, failures=0)
    await _seed(repo, "balanced", successes=0, failures=20)
    await learner.run_once()

    after = learner.snapshot()
    assert set(after["learned_overrides"]) == {"balanced"}
    assert set(after["learned_overrides"]["balanced"]) == {f.name for f in fields(PolicyWeights)}
