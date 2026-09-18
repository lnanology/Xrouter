"""Policy Learning (Phase 5, section 三十六): reads A/B Routing's own
ab_results data and nudges a losing policy's weights a step toward the
winning policy's weights, so a policy demonstrably underperforming in real
traffic gradually converges toward one that's winning.

This is deliberately a single callable "compute one adjustment step"
operation (run_once()), not its own scheduled background loop — the
repeated scheduling is Evolution Engine's job (Phase 5's next piece), which
will call run_once() on a timer with its own guardrails. Building a second
competing scheduler here would be the kind of unneeded infrastructure this
project's own rules forbid.

The fitness formula below is a plain, documented, deterministic scalar —
not a real ML model, and not claimed to be one. success_rate dominates by
construction; latency and quality are small tie-breaking terms."""
from __future__ import annotations

from dataclasses import fields, replace

from app.contracts.policy import PolicyWeights
from app.storage.repositories.metrics import MetricsRepository

MIN_WEIGHT = 0.05
MAX_WEIGHT = 5.0


def _clamp(value: float) -> float:
    return max(MIN_WEIGHT, min(MAX_WEIGHT, value))


def _nudge(current: PolicyWeights, target: PolicyWeights, learning_rate: float) -> PolicyWeights:
    updates = {}
    for f in fields(PolicyWeights):
        cur = getattr(current, f.name)
        tgt = getattr(target, f.name)
        updates[f.name] = _clamp(cur + learning_rate * (tgt - cur))
    return replace(current, **updates)


class PolicyLearner:
    def __init__(
        self, variants: list[str], metrics_repo: MetricsRepository, enabled: bool = False,
        min_samples: int = 20, learning_rate: float = 0.15, min_margin: float = 0.05,
        latency_weight_per_second: float = 0.05, quality_weight: float = 0.1,
    ):
        self._variants = list(variants)
        self._repo = metrics_repo
        self._enabled = enabled
        self._min_samples = min_samples
        self._learning_rate = learning_rate
        self._min_margin = min_margin
        self._latency_weight_per_second = latency_weight_per_second
        self._quality_weight = quality_weight
        self._learned: dict[str, PolicyWeights] = {}

    def weights_for(self, policy_key: str, base: PolicyWeights) -> PolicyWeights:
        """The read path real routing uses (app/routing/router.py's
        resolve_policy). Disabled -> always the static base weights,
        regardless of what may have been learned via a manual relearn --
        the same "disabled = complete no-op" precedent as ABRouter.assign().
        """
        if not self._enabled:
            return base
        return self._learned.get(policy_key, base)

    def fitness(self, row: dict) -> float:
        """Public: Evolution Engine (app/routing/evolution_engine.py)
        reuses this exact instance method to evaluate a nudge's post-nudge
        performance, so the nudge decision and the later rollback
        evaluation can never drift out of formula-sync with each other."""
        success_rate = row.get("success_rate") or 0.0
        avg_latency_ms = row.get("avg_latency_ms") or 0.0
        avg_quality_score = row.get("avg_quality_score") or 0.0
        return (
            success_rate
            + self._quality_weight * avg_quality_score
            - self._latency_weight_per_second * (avg_latency_ms / 1000.0)
        )

    async def run_once(self, exclude: set[str] | None = None) -> dict:
        """Computes (and, for any loser whose margin clears the threshold,
        stores) one adjustment step from the current ab_results. Always
        safe to call, even when disabled -- a disabled learner still
        computes and stores overrides here (a preview), it's weights_for()
        above that gates whether real routing actually uses them.

        `exclude`: policy keys that must never be nudged as a loser this
        call (still included in fitness/eligible so they can still serve
        as a winner reference for others). Evolution Engine passes in
        every policy it's still waiting to evaluate from a prior nudge --
        without this, the same policy could get nudged again before
        anyone checked whether the first nudge even helped."""
        exclude = exclude or set()
        summary = await self._repo.ab_summary()
        eligible = {
            v: summary[v] for v in self._variants
            if v in summary and summary[v].get("count", 0) >= self._min_samples
        }
        if len(eligible) < 2:
            return {
                "applied": False, "reason": "insufficient_samples",
                "eligible_variants": sorted(eligible), "min_samples": self._min_samples,
            }

        fitness = {v: self.fitness(row) for v, row in eligible.items()}
        winner = max(fitness, key=fitness.get)

        adjustments: dict[str, dict] = {}
        for loser in eligible:
            if loser == winner or loser in exclude:
                continue
            margin = fitness[winner] - fitness[loser]
            if margin < self._min_margin:
                continue
            from app.routing.router import POLICY_WEIGHTS  # local import: avoids a circular import at module load

            winner_current = self._learned.get(winner, POLICY_WEIGHTS.get(winner, POLICY_WEIGHTS["balanced"]))
            loser_base = POLICY_WEIGHTS.get(loser, POLICY_WEIGHTS["balanced"])
            prior_override = self._learned.get(loser)  # None means "no override existed before this call"
            loser_current = prior_override if prior_override is not None else loser_base

            new_weights = _nudge(loser_current, winner_current, self._learning_rate)
            self._learned[loser] = new_weights
            adjustments[loser] = {
                "toward": winner, "margin": margin, "fitness_before": fitness[loser],
                "previous_weights": (
                    {f.name: getattr(prior_override, f.name) for f in fields(PolicyWeights)}
                    if prior_override is not None else None
                ),
                "weights": {f.name: getattr(new_weights, f.name) for f in fields(PolicyWeights)},
            }

        return {"applied": bool(adjustments), "fitness": fitness, "adjustments": adjustments}

    def revert(self, policy_key: str, weights: dict | None) -> None:
        """Restores policy_key's learned override to a previous state --
        None removes any override entirely (weights_for() falls back to
        the static base again); a dict (the same shape run_once()'s
        adjustments[...]["previous_weights"]/["weights"] already return)
        reconstructs and restores that exact PolicyWeights. Used by
        Evolution Engine's rollback guardrail when a nudge's real-world
        post-nudge performance came in worse than before."""
        if weights is None:
            self._learned.pop(policy_key, None)
        else:
            self._learned[policy_key] = PolicyWeights(**weights)

    def snapshot(self) -> dict:
        return {
            "enabled": self._enabled,
            "variants": self._variants,
            "learned_overrides": {
                variant: {f.name: getattr(weights, f.name) for f in fields(PolicyWeights)}
                for variant, weights in self._learned.items()
            },
        }
