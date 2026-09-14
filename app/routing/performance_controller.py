"""Performance Controller (Phase 2, section 二十五): adjusts each provider's
routing weight from *observed* telemetry (rolling latency + success rate),
not just the static config-seeded scores. Guarded by a minimum sample count
so a handful of early requests can't send a provider's weight swinging —
avoiding the routing oscillation the spec explicitly warns about.

Design:
  - Every fallback attempt (success or failure) calls `record()`, updating
    an exponential moving average of latency and success rate per provider.
  - `weight_multiplier(provider_id)` returns 1.0 (neutral — no opinion yet)
    until MIN_SAMPLES is reached, then a value derived from those EMAs.
  - A lightweight background loop periodically persists a snapshot to the
    `routing_metrics` table (so trends are inspectable after the fact) and
    emits a `routing.changed` event when a provider's multiplier moves by
    more than a small threshold since the last snapshot — advisory only,
    routing itself always reads the live multiplier.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from app.observability.events import EventBus
from app.observability.logging import get_logger
from app.storage.repositories.metrics import MetricsRepository

logger = get_logger("performance_controller")

MIN_SAMPLES = 10
EMA_ALPHA = 0.2
TARGET_LATENCY_MS = 3000.0
MIN_LATENCY_COMPONENT = 0.3
CHANGE_EVENT_THRESHOLD = 0.15


@dataclass
class _ProviderPerf:
    sample_count: int = 0
    ema_latency_ms: float = TARGET_LATENCY_MS
    ema_success_rate: float = 1.0
    last_reported_multiplier: float = 1.0

    def record(self, latency_ms: float, success: bool) -> None:
        if self.sample_count == 0:
            self.ema_latency_ms = latency_ms
            self.ema_success_rate = 1.0 if success else 0.0
        else:
            self.ema_latency_ms = EMA_ALPHA * latency_ms + (1 - EMA_ALPHA) * self.ema_latency_ms
            outcome = 1.0 if success else 0.0
            self.ema_success_rate = EMA_ALPHA * outcome + (1 - EMA_ALPHA) * self.ema_success_rate
        self.sample_count += 1

    def multiplier(self) -> float:
        if self.sample_count < MIN_SAMPLES:
            return 1.0
        latency_component = max(TARGET_LATENCY_MS / max(self.ema_latency_ms, TARGET_LATENCY_MS), MIN_LATENCY_COMPONENT)
        return max(min(self.ema_success_rate * latency_component, 1.0), MIN_LATENCY_COMPONENT)


class PerformanceController:
    def __init__(
        self,
        metrics_repo: MetricsRepository | None = None,
        events: EventBus | None = None,
        snapshot_interval_seconds: float = 60.0,
    ):
        self._stats: dict[str, _ProviderPerf] = {}
        self._repo = metrics_repo
        self._events = events
        self._interval = snapshot_interval_seconds
        self._task: asyncio.Task | None = None
        self._stopping = asyncio.Event()

    def _get(self, provider_id: str) -> _ProviderPerf:
        if provider_id not in self._stats:
            self._stats[provider_id] = _ProviderPerf()
        return self._stats[provider_id]

    def record(self, provider_id: str, latency_ms: float, success: bool) -> None:
        self._get(provider_id).record(latency_ms, success)

    def weight_multiplier(self, provider_id: str) -> float:
        if provider_id not in self._stats:
            return 1.0
        return self._stats[provider_id].multiplier()

    def snapshot(self) -> dict:
        return {
            pid: {
                "sample_count": s.sample_count,
                "ema_latency_ms": round(s.ema_latency_ms, 1),
                "ema_success_rate": round(s.ema_success_rate, 3),
                "weight_multiplier": round(s.multiplier(), 3),
                "gated": s.sample_count < MIN_SAMPLES,
            }
            for pid, s in self._stats.items()
        }

    async def _snapshot_once(self) -> None:
        snap = self.snapshot()
        if self._repo is not None and snap:
            await self._repo.record_routing_metrics(snap)
        for pid, s in self._stats.items():
            current = s.multiplier()
            if abs(current - s.last_reported_multiplier) >= CHANGE_EVENT_THRESHOLD:
                if self._events is not None:
                    self._events.emit(
                        "routing.changed",
                        {"provider_id": pid, "previous_weight": s.last_reported_multiplier, "new_weight": current},
                    )
                s.last_reported_multiplier = current

    async def _loop(self) -> None:
        while not self._stopping.is_set():
            try:
                await self._snapshot_once()
            except Exception:
                logger.exception("performance controller snapshot failed")
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=self._interval)
            except asyncio.TimeoutError:
                pass

    def start(self) -> None:
        if self._task is None:
            self._stopping.clear()
            self._task = asyncio.create_task(self._loop())
            logger.info("performance controller started (interval=%.0fs, min_samples=%d)", self._interval, MIN_SAMPLES)

    async def stop(self) -> None:
        if self._task is not None:
            self._stopping.set()
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                self._task.cancel()
            self._task = None
            logger.info("performance controller stopped")
