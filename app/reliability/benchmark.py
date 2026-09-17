"""Automated Benchmark (Phase 5's first piece, spec 三十六). Turns the
probe already implemented for `POST /admin/benchmark`
(app/api/admin.py) into a scheduled, independently testable background
task -- the exact same hand-rolled loop shape app/reliability/health.py's
HealthMonitor and app/routing/performance_controller.py's
PerformanceController already use (start() spawns an asyncio.Task running
_loop(), which does one unit of work then waits up to `interval_seconds`
for a stop signal; stop() signals and cleanly tears the task down). No new
generic scheduler abstraction is introduced -- two prior instances of this
exact shape is this codebase's established convention for a periodic
background task, not a gap that needs filling.

Deliberately narrow in scope: this piece only makes benchmarking actually
automatic and its history actually readable back (`recent_benchmarks()`
already existed but was never exposed anywhere). It does not analyze
trends, does not feed results back into routing scores, and does not take
any remediating action -- that's Policy Learning's and Self-healing's job,
later Phase 5 pieces that don't exist yet. Building any of that here would
be exactly the "fake placeholder functionality" the project's rule 4
forbids: those later pieces don't exist yet to receive this data."""
from __future__ import annotations

import asyncio
import time

from app.core.registry import ModelRegistry, ProviderRegistry
from app.observability.logging import get_logger
from app.storage.repositories.metrics import MetricsRepository

logger = get_logger("benchmark")

DEFAULT_PROMPT = "Reply with the single word: OK"


class BenchmarkScheduler:
    def __init__(
        self,
        providers: ProviderRegistry,
        models: ModelRegistry,
        metrics_repo: MetricsRepository,
        prompt: str = DEFAULT_PROMPT,
        max_tokens: int = 8,
        interval_seconds: float = 3600.0,
    ):
        self._providers = providers
        self._models = models
        self._repo = metrics_repo
        self._prompt = prompt
        self._max_tokens = max_tokens
        self._interval = interval_seconds
        self._task: asyncio.Task | None = None
        self._stopping = asyncio.Event()

    async def run_once(self) -> list[dict]:
        """Sends one small prompt to each enabled provider's first
        available model and records latency/success -- the same probe
        `admin_benchmark` used to run inline, moved here verbatim so its
        behavior is provably unchanged and now independently testable."""
        from app.contracts.request import ChatCompletionRequest, ChatMessage

        results: list[dict] = []
        test_request = ChatCompletionRequest(
            model="auto", messages=[ChatMessage(role="user", content=self._prompt)],
            stream=False, max_tokens=self._max_tokens,
        )
        for pid, provider in self._providers.all().items():
            if not self._providers.is_enabled(pid):
                continue
            models = self._models.for_provider(pid)
            if not models:
                continue
            model = models[0]
            start = time.time()
            success = False
            try:
                await provider.chat(model.name, test_request)
                success = True
            except Exception as e:
                logger.info("benchmark call failed for %s/%s: %s", pid, model.name, e)
            latency_ms = (time.time() - start) * 1000
            await self._repo.record_benchmark(pid, model.name, None, latency_ms, None, success)
            results.append({"provider": pid, "model": model.name, "latency_ms": round(latency_ms, 1), "success": success})
        return results

    async def _loop(self) -> None:
        while not self._stopping.is_set():
            try:
                await self.run_once()
            except Exception:
                logger.exception("automated benchmark run failed")
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=self._interval)
            except asyncio.TimeoutError:
                pass

    def start(self) -> None:
        if self._task is None:
            self._stopping.clear()
            self._task = asyncio.create_task(self._loop())
            logger.info("automated benchmark scheduler started (interval=%.0fs)", self._interval)

    async def stop(self) -> None:
        if self._task is not None:
            self._stopping.set()
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                self._task.cancel()
            self._task = None
            logger.info("automated benchmark scheduler stopped")
