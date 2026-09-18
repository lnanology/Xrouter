"""Evolution Engine (Phase 5's fourth piece, section 三十六): the loop that
repeatedly asks PolicyLearner for a fresh nudge, and -- this is the part
PolicyLearner deliberately doesn't do itself -- checks whether the nudge it
made *last* cycle actually helped once real traffic ran under the new
weights, rolling it back if it didn't. Genuine "variation, evaluation,
retention-or-reversion", not a fake wrapper that just calls run_once() on a
timer with no real evaluation step.

Same hand-rolled start()/_loop()/stop() shape app/reliability/health.py's
HealthMonitor, app/routing/performance_controller.py's PerformanceController,
and app/reliability/benchmark.py's BenchmarkScheduler already use -- the
fourth instance of this exact shape, not a new scheduler abstraction.

Evaluation is an honest pre/post comparison on the same policy across time
(fitness right before the nudge vs. fitness over fresh outcomes recorded
after it), not a rigorous causal test -- real-world traffic can drift for
unrelated reasons too. That limitation is real and worth stating plainly
rather than glossing over, the same way A/B Routing's anonymous-caller
limitation is documented rather than hidden."""
from __future__ import annotations

import asyncio
import time

from app.observability.logging import get_logger
from app.routing.policy_learner import PolicyLearner
from app.storage.repositories.metrics import MetricsRepository

logger = get_logger("evolution_engine")


class EvolutionEngine:
    def __init__(
        self, policy_learner: PolicyLearner, metrics_repo: MetricsRepository, enabled: bool = False,
        interval_seconds: float = 3600.0, evaluation_samples: int = 20, rollback_tolerance: float = 0.02,
    ):
        self._policy_learner = policy_learner
        self._repo = metrics_repo
        self._enabled = enabled
        self._interval = interval_seconds
        self._evaluation_samples = evaluation_samples
        self._rollback_tolerance = rollback_tolerance
        # policy_key -> {"applied_at": float, "previous_weights": dict | None, "baseline_fitness": float}
        self._pending: dict[str, dict] = {}
        self._task: asyncio.Task | None = None
        self._stopping = asyncio.Event()

    async def _evaluate_pending(self) -> dict:
        """Resolves every currently-pending nudge it has enough fresh
        evidence for; leaves the rest pending untouched (not enough new
        samples yet -- tries again next cycle)."""
        results: dict[str, dict] = {}
        for policy_key, pending in list(self._pending.items()):
            summary_since = await self._repo.ab_summary(since=pending["applied_at"])
            row = summary_since.get(policy_key)
            if row is None or row.get("count", 0) < self._evaluation_samples:
                continue  # not enough fresh evidence yet -- leave pending

            new_fitness = self._policy_learner.fitness(row)
            baseline = pending["baseline_fitness"]
            if new_fitness < baseline - self._rollback_tolerance:
                self._policy_learner.revert(policy_key, pending["previous_weights"])
                outcome = "reverted"
            else:
                outcome = "confirmed"
            results[policy_key] = {"outcome": outcome, "new_fitness": new_fitness, "baseline_fitness": baseline}
            del self._pending[policy_key]
        return results

    async def run_once(self) -> dict:
        evaluated = await self._evaluate_pending()

        learn_result = await self._policy_learner.run_once(exclude=set(self._pending))

        now = time.time()
        for policy_key, adjustment in learn_result.get("adjustments", {}).items():
            self._pending[policy_key] = {
                "applied_at": now,
                "previous_weights": adjustment["previous_weights"],
                "baseline_fitness": adjustment["fitness_before"],
            }

        return {"evaluated": evaluated, "learn": learn_result, "pending": sorted(self._pending)}

    async def _loop(self) -> None:
        while not self._stopping.is_set():
            try:
                await self.run_once()
            except Exception:
                logger.exception("evolution engine cycle failed")
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=self._interval)
            except asyncio.TimeoutError:
                pass

    def start(self) -> None:
        if self._task is None:
            self._stopping.clear()
            self._task = asyncio.create_task(self._loop())
            logger.info("evolution engine started (interval=%.0fs)", self._interval)

    async def stop(self) -> None:
        if self._task is not None:
            self._stopping.set()
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                self._task.cancel()
            self._task = None
            logger.info("evolution engine stopped")

    def snapshot(self) -> dict:
        return {"enabled": self._enabled, "interval_seconds": self._interval, "pending": self._pending}
