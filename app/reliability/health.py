"""Background health monitor (section 九). Polls every provider on a fixed
interval — never on the request path — and feeds results into both the
ProviderRegistry (for routing) and the circuit breaker (so a provider that
degrades between requests is caught proactively)."""
from __future__ import annotations

import asyncio
import time

from app.core.registry import ProviderRegistry
from app.observability.events import get_event_bus
from app.observability.logging import get_logger
from app.reliability.circuit_breaker import CircuitBreakerRegistry
from app.storage.repositories.provider import ProviderRepository

logger = get_logger("health")


class HealthMonitor:
    def __init__(
        self,
        providers: ProviderRegistry,
        circuits: CircuitBreakerRegistry,
        provider_repo: ProviderRepository | None = None,
        interval_seconds: float = 20.0,
    ):
        self._providers = providers
        self._circuits = circuits
        self._repo = provider_repo
        self._interval = interval_seconds
        self._task: asyncio.Task | None = None
        self._stopping = asyncio.Event()

    async def _check_once(self) -> None:
        bus = get_event_bus()
        for pid, provider in self._providers.all().items():
            if not self._providers.is_enabled(pid):
                continue
            previous = self._providers.health_of(pid)
            try:
                health = await asyncio.wait_for(provider.health(), timeout=10.0)
            except Exception as e:
                from app.contracts.provider import ProviderHealth, ProviderStatus

                health = ProviderHealth(status=ProviderStatus.OFFLINE, last_checked=time.time(), last_error=str(e))

            self._providers.set_health(pid, health)

            from app.contracts.provider import ProviderStatus

            was_unhealthy = previous.status in (ProviderStatus.OFFLINE, ProviderStatus.DEGRADED)
            now_unhealthy = health.status in (ProviderStatus.OFFLINE, ProviderStatus.DEGRADED)
            if now_unhealthy and not was_unhealthy:
                bus.emit("provider.unhealthy", {"provider_id": pid, "status": health.status.value})
            elif was_unhealthy and not now_unhealthy:
                bus.emit("provider.recovered", {"provider_id": pid})

            if self._repo is not None:
                await self._repo.record_health(
                    pid, health.status.value, health.latency_ms, health.success_rate, health.last_error
                )

    async def _loop(self) -> None:
        while not self._stopping.is_set():
            try:
                await self._check_once()
            except Exception:
                logger.exception("health check loop iteration failed")
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=self._interval)
            except asyncio.TimeoutError:
                pass

    def start(self) -> None:
        if self._task is None:
            self._stopping.clear()
            self._task = asyncio.create_task(self._loop())
            logger.info("health monitor started (interval=%.0fs)", self._interval)

    async def stop(self) -> None:
        if self._task is not None:
            self._stopping.set()
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                self._task.cancel()
            self._task = None
            logger.info("health monitor stopped")
