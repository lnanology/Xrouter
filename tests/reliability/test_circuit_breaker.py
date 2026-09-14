import time

from app.reliability.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState


def test_closes_to_open_after_threshold():
    cb = CircuitBreaker("p1", CircuitBreakerConfig(failure_threshold=3, cooldown_seconds=0.1))
    assert cb.state == CircuitState.CLOSED
    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitState.CLOSED
    cb.record_failure()
    assert cb.state == CircuitState.OPEN


def test_open_blocks_requests_until_cooldown():
    cb = CircuitBreaker("p1", CircuitBreakerConfig(failure_threshold=1, cooldown_seconds=0.05))
    cb.record_failure()
    assert cb.state == CircuitState.OPEN
    assert cb.allow_request() is False
    time.sleep(0.06)
    assert cb.allow_request() is True
    assert cb.state == CircuitState.HALF_OPEN


def test_half_open_success_closes():
    cb = CircuitBreaker("p1", CircuitBreakerConfig(failure_threshold=1, cooldown_seconds=0.01))
    cb.record_failure()
    time.sleep(0.02)
    assert cb.allow_request() is True
    cb.record_success()
    assert cb.state == CircuitState.CLOSED
    assert cb.consecutive_failures == 0


def test_half_open_failure_reopens_with_backoff():
    cb = CircuitBreaker("p1", CircuitBreakerConfig(failure_threshold=1, cooldown_seconds=0.01, max_cooldown_seconds=1.0))
    cb.record_failure()
    time.sleep(0.02)
    assert cb.allow_request() is True
    cb.record_failure()
    assert cb.state == CircuitState.OPEN
    assert cb.current_cooldown > 0.01  # backed off


def test_force_open_admin_override():
    cb = CircuitBreaker("p1")
    assert cb.state == CircuitState.CLOSED
    cb.force_open(cooldown_seconds=100)
    assert cb.state == CircuitState.OPEN
    assert cb.allow_request() is False
