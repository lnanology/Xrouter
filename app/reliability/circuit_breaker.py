"""Per-provider circuit breaker. CLOSED -> OPEN -> HALF_OPEN -> CLOSED.

Core rule: after `failure_threshold` consecutive failures the circuit trips
OPEN and the provider is not selected by the router. After `cooldown_seconds`
it moves to HALF_OPEN, allowing exactly one trial request; success closes it,
failure re-opens it (with backoff on the cooldown)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 3
    cooldown_seconds: float = 30.0
    max_cooldown_seconds: float = 300.0
    half_open_max_probes: int = 1


class CircuitBreaker:
    def __init__(self, key: str, config: CircuitBreakerConfig | None = None):
        self.key = key
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self.opened_at: float | None = None
        self.current_cooldown = self.config.cooldown_seconds
        self._half_open_probes_in_flight = 0

    def allow_request(self) -> bool:
        now = time.time()
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if self.opened_at is not None and (now - self.opened_at) >= self.current_cooldown:
                self.state = CircuitState.HALF_OPEN
                self._half_open_probes_in_flight = 0
            else:
                return False
        if self.state == CircuitState.HALF_OPEN:
            if self._half_open_probes_in_flight < self.config.half_open_max_probes:
                self._half_open_probes_in_flight += 1
                return True
            return False
        return False

    def record_success(self) -> None:
        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self.opened_at = None
        self.current_cooldown = self.config.cooldown_seconds
        self._half_open_probes_in_flight = 0

    def force_open(self, cooldown_seconds: float | None = None) -> None:
        """Administrative override (POST /admin/providers/{id}/cooldown) —
        immediately trips the breaker without waiting for failure_threshold."""
        self.state = CircuitState.OPEN
        self.opened_at = time.time()
        self.current_cooldown = cooldown_seconds if cooldown_seconds is not None else self.config.cooldown_seconds

    def record_failure(self) -> None:
        if self.state == CircuitState.HALF_OPEN:
            # Failed probe: reopen with a longer cooldown (bounded).
            self.state = CircuitState.OPEN
            self.opened_at = time.time()
            self.current_cooldown = min(self.current_cooldown * 2, self.config.max_cooldown_seconds)
            self._half_open_probes_in_flight = 0
            return

        self.consecutive_failures += 1
        if self.consecutive_failures >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            self.opened_at = time.time()

    def snapshot(self) -> dict:
        return {
            "key": self.key,
            "state": self.state.value,
            "consecutive_failures": self.consecutive_failures,
            "opened_at": self.opened_at,
            "cooldown_seconds": self.current_cooldown,
        }


class CircuitBreakerRegistry:
    """Keeps one breaker per provider (keyed by provider_id)."""

    def __init__(self, config: CircuitBreakerConfig | None = None):
        self._config = config or CircuitBreakerConfig()
        self._breakers: dict[str, CircuitBreaker] = {}

    def get(self, key: str) -> CircuitBreaker:
        if key not in self._breakers:
            self._breakers[key] = CircuitBreaker(key, self._config)
        return self._breakers[key]

    def all(self) -> dict[str, CircuitBreaker]:
        return dict(self._breakers)
