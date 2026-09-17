"""Engine-level A/B Routing tests (Phase 5's second piece), through the real
ChatEngine.handle_chat() pipeline (cache -> route -> execute -> telemetry) --
tests/routing/test_ab_router.py covers ABRouter's own assignment/aggregation
logic in isolation; this file covers how ChatEngine wires it in."""
from __future__ import annotations

import pytest

from app.cache.manager import CacheManager
from app.contracts.request import ChatCompletionRequest, ChatMessage
from app.core.config import CacheConfig
from app.core.errors import NoAvailableModelError
from tests.helpers import build_test_engine

VARIANTS = ["quality", "balanced"]


def _req(content: str = "hi", routing_policy: str | None = None, user: str | None = None) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="auto", messages=[ChatMessage(role="user", content=content)],
        routing_policy=routing_policy, user=user,
    )


@pytest.mark.asyncio
async def test_explicit_routing_policy_bypasses_ab_entirely_even_when_it_matches_a_variant_name(tmp_path):
    engine = await build_test_engine(
        tmp_path, {"solo": {"content": "ok"}}, ab_routing_overrides={"enabled": True, "variants": VARIANTS},
    )
    response = await engine.handle_chat(_req(routing_policy="quality"))

    assert not (response.xrouter or {}).get("ab_experiment")
    assert await engine.ctx.metrics_repo.ab_summary() == {}


@pytest.mark.asyncio
async def test_disabled_ab_routing_is_a_complete_noop(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"content": "ok"}})  # ab_routing defaults to disabled
    response = await engine.handle_chat(_req())

    assert not (response.xrouter or {}).get("ab_experiment")
    assert await engine.ctx.metrics_repo.ab_summary() == {}


@pytest.mark.asyncio
async def test_enabled_ab_routing_assigns_tags_and_persists_an_outcome(tmp_path):
    engine = await build_test_engine(
        tmp_path, {"solo": {"content": "ok"}}, ab_routing_overrides={"enabled": True, "variants": VARIANTS},
    )
    response = await engine.handle_chat(_req(user="some-user"))

    assert response.xrouter["ab_experiment"] is True
    assigned = response.xrouter["policy"]
    assert assigned in VARIANTS

    summary = await engine.ctx.metrics_repo.ab_summary()
    assert summary[assigned]["count"] == 1
    assert summary[assigned]["success_rate"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_a_no_available_model_failure_still_records_an_ab_outcome(tmp_path):
    engine = await build_test_engine(
        tmp_path, {"broken": {"behavior": "server_error"}},
        ab_routing_overrides={"enabled": True, "variants": VARIANTS},
    )
    with pytest.raises(NoAvailableModelError):
        await engine.handle_chat(_req(user="some-user"))

    summary = await engine.ctx.metrics_repo.ab_summary()
    # Exactly one variant was assigned (sticky on "some-user") and its one
    # recorded outcome must be a failure.
    assert sum(v["count"] for v in summary.values()) == 1
    assert all(v["success_rate"] == pytest.approx(0.0) for v in summary.values() if v["count"])


@pytest.mark.asyncio
async def test_a_cache_hit_does_not_record_a_second_ab_outcome(tmp_path):
    engine = await build_test_engine(
        tmp_path, {"solo": {"content": "ok"}}, ab_routing_overrides={"enabled": True, "variants": VARIANTS},
    )
    # build_test_engine hardcodes cache disabled (it's not this piece's
    # concern) -- swap in a real, enabled CacheManager just for this test so
    # the second call can actually be served from cache. It must point at
    # the same "test.sqlite3" build_test_engine already initialized (the
    # cache_entries table only exists there, per app/cache/sqlite.py's own
    # "same database used for telemetry" design) rather than a fresh path.
    engine.ctx.cache = CacheManager(CacheConfig(enabled=True), str(tmp_path / "test.sqlite3"))

    req = _req(content="cache me please", user="sticky-user")
    first = await engine.handle_chat(req)
    assert first.xrouter.get("ab_experiment") is True

    before = await engine.ctx.metrics_repo.ab_summary()
    total_before = sum(v["count"] for v in before.values())

    second = await engine.handle_chat(req)
    after = await engine.ctx.metrics_repo.ab_summary()
    total_after = sum(v["count"] for v in after.values())

    # The second call was a cache hit (same request, same sticky user) -- no
    # new ab_results row should have been added for it.
    assert total_after == total_before
