"""Self-healing (Phase 5's fifth and last piece): correlates two signals
that already exist for free -- ProviderRegistry.health_of() (updated by
HealthMonitor's own polling) and CircuitBreakerRegistry's per-provider
state (updated by real request traffic) -- and takes the one corrective
action nothing else in this codebase takes automatically:
ProviderRegistry.set_enabled().

This piece never re-probes a provider itself (that would duplicate
HealthMonitor's job -- exactly the unneeded infrastructure this project's
own rules forbid). It only reads state HealthMonitor and the circuit
breaker have already computed, and correlates it: a provider that's
looked unhealthy (OFFLINE/DEGRADED) or circuit-OPEN for `confirm_cycles`
consecutive checks gets auto-disabled; one that's looked healthy AND
circuit-CLOSED for `confirm_cycles` consecutive checks while
self-disabled gets auto re-enabled. A single good (or bad) tick resets
the opposite streak -- no leaky partial credit either direction.

Same hand-rolled start()/_loop()/stop() shape app/reliability/health.py's
HealthMonitor, app/routing/performance_controller.py's
PerformanceController, app/reliability/benchmark.py's BenchmarkScheduler,
and app/routing/evolution_engine.py's EvolutionEngine already use -- the
fifth instance of this exact shape, not a new scheduler abstraction.

Crucially, this never touches a provider an admin disabled directly:
recovery candidates are drawn only from `_disabled_by_self`, which only
ever gains an entry via this module's own auto-disable action. Every
provider-mutating admin route (enable/disable/cooldown in
app/api/admin.py) calls forget() right after it acts, so an explicit
admin decision always hands control back to the operator -- see forget()
below for the exact reasoning."""
from __future__ import annotations

import asyncio
import time

from app.contracts.provider import ProviderStatus
from app.core.registry import ProviderRegistry
from app.observability.events import EventBus
from app.observability.logging import get_logger
from app.reliability.circuit_breaker import CircuitBreakerRegistry, CircuitState

logger = get_logger("self_healer")

_UNHEALTHY_STATUSES = (ProviderStatus.OFFLINE, ProviderStatus.DEGRADED)


class SelfHealer:
    def __init__(
        self, providers: ProviderRegistry, circuits: CircuitBreakerRegistry, events: EventBus,
        enabled: bool = False, interval_seconds: float = 30.0, confirm_cycles: int = 3,
    ):
        self._providers = providers
        self._circuits = circuits
        self._events = events
        self._enabled = enabled
        self._interval = interval_seconds
        self._confirm_cycles = confirm_cycles
        self._bad_streak: dict[str, int] = {}
        self._good_streak: dict[str, int] = {}
        # provider_id -> {"disabled_at": float} -- ONLY providers this
        # module itself disabled. A provider an admin disabled directly
        # never enters this dict, so it's never a recovery candidate.
        self._disabled_by_self: dict[str, dict] = {}
        self._task: asyncio.Task | None = None
        self._stopping = asyncio.Event()

    def _is_unhealthy(self, provider_id: str) -> bool:
        health = self._providers.health_of(provider_id)
        breaker = self._circuits.get(provider_id)
        return health.status in _UNHEALTHY_STATUSES or breaker.state == CircuitState.OPEN

    async def run_once(self) -> dict:
        actions: dict[str, str] = {}

        # Disable candidates: every provider currently enabled (mirrors
        # HealthMonitor._check_once()'s own loop shape exactly).
        for pid in self._providers.all():
            if not self._providers.is_enabled(pid):
                continue
            if self._is_unhealthy(pid):
                self._bad_streak[pid] = self._bad_streak.get(pid, 0) + 1
                self._good_streak.pop(pid, None)
                if self._bad_streak[pid] >= self._confirm_cycles:
                    self._providers.set_enabled(pid, False)
                    self._disabled_by_self[pid] = {"disabled_at": time.time()}
                    self._bad_streak.pop(pid, None)
                    self._events.emit("self_healing.disabled", {"provider_id": pid})
                    actions[pid] = "disabled"
                    logger.warning("self-healing disabled provider '%s' after %d bad checks", pid, self._confirm_cycles)
            else:
                self._bad_streak.pop(pid, None)

        # Recovery candidates: only providers this module itself disabled.
        for pid in list(self._disabled_by_self):
            if self._is_unhealthy(pid):
                self._good_streak.pop(pid, None)
                continue
            self._good_streak[pid] = self._good_streak.get(pid, 0) + 1
            if self._good_streak[pid] >= self._confirm_cycles:
                self._providers.set_enabled(pid, True)
                del self._disabled_by_self[pid]
                self._good_streak.pop(pid, None)
                self._events.emit("self_healing.recovered", {"provider_id": pid})
                actions[pid] = "recovered"
                logger.info("self-healing re-enabled provider '%s' after %d good checks", pid, self._confirm_cycles)

        return {"actions": actions, "managed": sorted(self._disabled_by_self)}

    def forget(self, provider_id: str) -> None:
        """Called by every provider-mutating admin route (enable/disable/
        cooldown) right after it acts: an explicit admin action always
        hands control back to the operator. Clears this provider's
        streaks and _disabled_by_self membership so Self-healing never
        "corrects" what the admin just did on purpose -- an admin
        disabling a provider Self-healing had auto-disabled and wanting
        it to STAY off drops it from _disabled_by_self (never auto
        re-enabled again); an admin re-enabling a provider clears stale
        streak counts so a still-unhealthy provider needs a full FRESH
        confirm_cycles streak again, not an instant re-disable from
        leftover history."""
        self._disabled_by_self.pop(provider_id, None)
        self._bad_streak.pop(provider_id, None)
        self._good_streak.pop(provider_id, None)

    async def _loop(self) -> None:
        while not self._stopping.is_set():
            try:
                await self.run_once()
            except Exception:
                logger.exception("self-healing cycle failed")
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=self._interval)
            except asyncio.TimeoutError:
                pass

    def start(self) -> None:
        if self._task is None:
            self._stopping.clear()
            self._task = asyncio.create_task(self._loop())
            logger.info("self-healer started (interval=%.0fs, confirm_cycles=%d)", self._interval, self._confirm_cycles)

    async def stop(self) -> None:
        if self._task is not None:
            self._stopping.set()
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                self._task.cancel()
            self._task = None
            logger.info("self-healer stopped")

    def snapshot(self) -> dict:
        return {
            "enabled": self._enabled, "interval_seconds": self._interval, "confirm_cycles": self._confirm_cycles,
            "disabled_by_self_healing": {pid: meta["disabled_at"] for pid, meta in self._disabled_by_self.items()},
        }
