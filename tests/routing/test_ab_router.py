"""Pure unit tests for ABRouter (Phase 5's second piece) -- assignment logic
and outcome persistence/aggregation, independent of ChatEngine. Engine-level
wiring (cache-hit exclusion, ab_experiment tagging, telemetry threading) is
covered separately in tests/unit/test_engine_ab_routing.py."""
from __future__ import annotations

import pytest

from app.contracts.request import ChatCompletionRequest, ChatMessage
from app.routing.ab_router import ABRouter
from app.storage.database import Database
from app.storage.repositories.metrics import MetricsRepository


async def _build(tmp_path, variants: list[str], enabled: bool = True) -> tuple[ABRouter, MetricsRepository]:
    db = Database(str(tmp_path / "test.sqlite3"))
    await db.init()
    metrics_repo = MetricsRepository(db)
    return ABRouter(variants, metrics_repo, enabled=enabled), metrics_repo


def _req(user: str | None = None) -> ChatCompletionRequest:
    return ChatCompletionRequest(model="auto", messages=[ChatMessage(role="user", content="hi")], user=user)


# --- assign() ----------------------------------------------------------

@pytest.mark.asyncio
async def test_assign_returns_none_when_disabled(tmp_path):
    router, _ = await _build(tmp_path, ["quality", "balanced"], enabled=False)
    assert router.assign(_req(user="u1")) is None


@pytest.mark.asyncio
async def test_assign_returns_none_with_fewer_than_two_variants(tmp_path):
    router_one, _ = await _build(tmp_path, ["only"], enabled=True)
    assert router_one.assign(_req(user="u1")) is None

    router_none, _ = await _build(tmp_path, [], enabled=True)
    assert router_none.assign(_req(user="u1")) is None


@pytest.mark.asyncio
async def test_assign_is_sticky_for_the_same_user(tmp_path):
    router, _ = await _build(tmp_path, ["quality", "balanced", "fastest"], enabled=True)
    req = _req(user="same-user-every-time")
    first = router.assign(req)
    assert first in {"quality", "balanced", "fastest"}
    for _ in range(20):
        assert router.assign(req) == first


@pytest.mark.asyncio
async def test_assign_differs_across_users_at_least_sometimes(tmp_path):
    # Not a strict requirement of hashing, but with 3 variants and 30 distinct
    # users it would be exceptionally unlucky for every single user to land
    # on the same variant -- a real smoke check that assignment is actually
    # keyed on the user, not hardcoded to always return variants[0].
    router, _ = await _build(tmp_path, ["quality", "balanced", "fastest"], enabled=True)
    seen = {router.assign(_req(user=f"user-{i}")) for i in range(30)}
    assert len(seen) > 1


@pytest.mark.asyncio
async def test_assign_distributes_roughly_evenly_across_anonymous_requests(tmp_path):
    # No `user` on the request -> each call falls back to a fresh uuid4, so
    # across many anonymous requests the split should still be roughly even
    # in aggregate (generous tolerance -- this is a statistical check, not a
    # precise 50/50 guarantee).
    router, _ = await _build(tmp_path, ["quality", "balanced"], enabled=True)
    counts = {"quality": 0, "balanced": 0}
    for _ in range(400):
        variant = router.assign(_req())
        counts[variant] += 1
    assert 120 < counts["quality"] < 280
    assert 120 < counts["balanced"] < 280


# --- record_outcome() / MetricsRepository.ab_summary() -----------------

@pytest.mark.asyncio
async def test_record_outcome_persists_and_ab_summary_aggregates_per_variant(tmp_path):
    router, metrics_repo = await _build(tmp_path, ["quality", "balanced"], enabled=True)

    await router.record_outcome("quality", success=True, latency_ms=100.0, quality_score=0.9)
    await router.record_outcome("quality", success=False, latency_ms=200.0, quality_score=None)
    await router.record_outcome("balanced", success=True, latency_ms=50.0, quality_score=0.7)

    summary = await metrics_repo.ab_summary()

    assert summary["quality"]["count"] == 2
    assert summary["quality"]["success_rate"] == pytest.approx(0.5)
    assert summary["quality"]["avg_latency_ms"] == pytest.approx(150.0)
    # AVG() over a nullable column skips NULLs in SQLite -- the one row with
    # quality_score=None doesn't drag this average down or turn it NULL.
    assert summary["quality"]["avg_quality_score"] == pytest.approx(0.9)

    assert summary["balanced"]["count"] == 1
    assert summary["balanced"]["success_rate"] == pytest.approx(1.0)
    assert summary["balanced"]["avg_latency_ms"] == pytest.approx(50.0)
    assert summary["balanced"]["avg_quality_score"] == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_ab_summary_with_no_recorded_outcomes_is_empty(tmp_path):
    _, metrics_repo = await _build(tmp_path, ["quality", "balanced"], enabled=True)
    assert await metrics_repo.ab_summary() == {}
