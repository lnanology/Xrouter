import pytest

from app.routing.performance_controller import MIN_SAMPLES, PerformanceController


def test_neutral_before_min_samples():
    pc = PerformanceController()
    for _ in range(MIN_SAMPLES - 1):
        pc.record("p1", latency_ms=10000, success=False)  # terrible, but not enough samples yet
    assert pc.weight_multiplier("p1") == 1.0


def test_unknown_provider_is_neutral():
    pc = PerformanceController()
    assert pc.weight_multiplier("never-seen") == 1.0


def test_consistently_fast_reliable_provider_gets_high_multiplier():
    pc = PerformanceController()
    for _ in range(MIN_SAMPLES + 5):
        pc.record("fast", latency_ms=200, success=True)
    assert pc.weight_multiplier("fast") > 0.9


def test_slow_provider_gets_penalized_but_not_zeroed():
    pc = PerformanceController()
    for _ in range(MIN_SAMPLES + 5):
        pc.record("slow", latency_ms=30000, success=True)
    mult = pc.weight_multiplier("slow")
    assert 0.0 < mult < 0.5


def test_unreliable_provider_penalized_via_success_rate():
    pc = PerformanceController()
    for i in range(MIN_SAMPLES + 10):
        pc.record("flaky", latency_ms=500, success=(i % 2 == 0))
    mult = pc.weight_multiplier("flaky")
    assert mult < 0.8


def test_gated_flag_in_snapshot():
    pc = PerformanceController()
    pc.record("p1", latency_ms=500, success=True)
    snap = pc.snapshot()
    assert snap["p1"]["gated"] is True
    assert snap["p1"]["sample_count"] == 1


def test_recovering_provider_improves_over_time():
    pc = PerformanceController()
    for _ in range(MIN_SAMPLES + 5):
        pc.record("p1", latency_ms=200, success=False)
    bad = pc.weight_multiplier("p1")
    for _ in range(30):
        pc.record("p1", latency_ms=200, success=True)
    good = pc.weight_multiplier("p1")
    assert good > bad


@pytest.mark.asyncio
async def test_start_stop_background_loop_graceful():
    pc = PerformanceController(snapshot_interval_seconds=0.05)
    pc.start()
    pc.record("p1", latency_ms=100, success=True)
    await pc.stop()  # must not hang or raise
