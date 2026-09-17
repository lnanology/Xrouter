"""Adaptive router (section 八). Builds a ranked, filtered candidate list
across every known model and turns it into a RoutingDecision (primary +
ordered fallback chain). No provider-specific branching lives here — it
only ever reads contracts.model.ModelInfo / contracts.provider.ProviderHealth.
"""
from __future__ import annotations

from app.contracts.policy import PolicyWeights
from app.contracts.request import ChatCompletionRequest
from app.contracts.router import RoutingCandidate, RoutingDecision
from app.core.errors import NoAvailableModelError
from app.core.registry import ModelRegistry, ProviderRegistry
from app.quota.tracker import QuotaTracker
from app.reliability.circuit_breaker import CircuitBreakerRegistry
from app.routing import policies as _policies_pkg  # noqa: F401  (documents location)
from app.routing.performance_controller import PerformanceController
from app.routing.policies import balanced, cheapest, fastest, quality, quota_aware, reliable
from app.routing.policy_learner import PolicyLearner
from app.routing.scorer import ScoreInput, score_candidate

# Public: app/routing/policy_learner.py reads this as the base weights a
# learned override is nudged from/toward.
POLICY_WEIGHTS: dict[str, PolicyWeights] = {
    "balanced": balanced.WEIGHTS,
    "fastest": fastest.WEIGHTS,
    "cheapest": cheapest.WEIGHTS,
    "reliable": reliable.WEIGHTS,
    "quota_aware": quota_aware.WEIGHTS,
    "quality": quality.WEIGHTS,
}

LOCAL_PROVIDER_TYPES = {"ollama"}


class AdaptiveRouter:
    def __init__(
        self,
        provider_registry: ProviderRegistry,
        model_registry: ModelRegistry,
        circuit_breakers: CircuitBreakerRegistry,
        quota_tracker: QuotaTracker,
        default_policy: str = "balanced",
        performance_controller: PerformanceController | None = None,
        policy_learner: PolicyLearner | None = None,
    ):
        self._providers = provider_registry
        self._models = model_registry
        self._circuits = circuit_breakers
        self._quota = quota_tracker
        self._default_policy = default_policy
        # Optional: a router built without one (e.g. in unit tests) simply
        # never applies a performance weight — every candidate stays neutral.
        self._performance = performance_controller
        # Optional: a router built without one (e.g. in unit tests) simply
        # never applies a learned override — every policy stays at its
        # static base weights, the same "no collaborator = neutral" shape
        # performance_controller already uses.
        self._policy_learner = policy_learner

    def resolve_policy(self, name: str | None) -> tuple[str, PolicyWeights]:
        key = (name or self._default_policy).lower()
        base = POLICY_WEIGHTS.get(key, POLICY_WEIGHTS["balanced"])
        weights = self._policy_learner.weights_for(key, base) if self._policy_learner else base
        return key, weights

    def _requested_model_filter(self, requested: str) -> str | None:
        """If the client asked for a specific known public_id
        ('ollama/llama3'), return it; 'auto'/unknown -> None (open choice)."""
        if not requested or requested in ("auto", "xrouter/auto", "xrouter-auto"):
            return None
        if self._models.get(requested) is not None:
            return requested
        return None

    def select(self, request: ChatCompletionRequest, policy_name: str | None = None) -> RoutingDecision:
        policy_key, weights = self.resolve_policy(policy_name or request.routing_policy)
        pinned = self._requested_model_filter(request.model)

        candidates: list[RoutingCandidate] = []
        pool = [self._models.get(pinned)] if pinned else self._models.all()

        for model in pool:
            if model is None or not model.active:
                continue
            if not self._providers.is_enabled(model.provider_id):
                continue
            provider = self._providers.get(model.provider_id)
            if provider is None:
                continue

            health = self._providers.health_of(model.provider_id)
            circuit = self._circuits.get(model.provider_id)
            quota_risk = self._quota.risk(model.provider_id)
            is_local = provider.config.type in LOCAL_PROVIDER_TYPES
            performance_weight = self._performance.weight_multiplier(model.provider_id) if self._performance else 1.0

            score = score_candidate(
                ScoreInput(
                    model=model,
                    provider_health=health,
                    circuit_state=circuit.state,
                    quota_risk=quota_risk,
                    requires_streaming=request.stream,
                    requires_tools=bool(request.tools),
                    performance_weight=performance_weight,
                ),
                weights,
                is_local,
            )
            if score is None:
                continue
            candidates.append(RoutingCandidate(provider_id=model.provider_id, model_id=model.name, score=score))

        if not candidates:
            raise NoAvailableModelError(
                "No healthy provider/model available for this request "
                "(all disabled, unhealthy, circuit-open, or quota-critical)."
            )

        candidates.sort(key=lambda c: c.score, reverse=True)
        return RoutingDecision(primary=candidates[0], fallback_chain=candidates[1:], policy=policy_key)
