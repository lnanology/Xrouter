import pytest

from app.contracts.policy import PolicyWeights
from app.contracts.provider import ProviderConfig, ProviderHealth, ProviderStatus
from app.contracts.request import ChatCompletionRequest, ChatMessage
from app.core.errors import NoAvailableModelError
from app.core.registry import ModelRegistry, ProviderRegistry
from app.quota.tracker import QuotaTracker
from app.reliability.circuit_breaker import CircuitBreakerRegistry, CircuitState
from app.routing.router import POLICY_WEIGHTS, AdaptiveRouter
from tests.helpers import FakeProvider, make_model


def _build(models_by_provider: dict[str, list], local_ids: set[str] | None = None):
    providers = ProviderRegistry()
    models = ModelRegistry()
    for pid, model_list in models_by_provider.items():
        cfg = ProviderConfig(id=pid, name=pid, type="ollama" if local_ids and pid in local_ids else "openai_compatible")
        fake = FakeProvider(cfg, model_list)
        providers.register(fake, cfg)
        for m in model_list:
            models._models[m.public_id()] = m
    circuits = CircuitBreakerRegistry()
    quota = QuotaTracker()
    return AdaptiveRouter(providers, models, circuits, quota), providers, circuits, quota


def _req():
    return ChatCompletionRequest(model="auto", messages=[ChatMessage(role="user", content="hi")])


def test_selects_best_scoring_candidate():
    weak = make_model("p1", "weak", quality_score=0.2, speed_score=0.2, reliability_score=0.2, cost_score=0.2)
    strong = make_model("p2", "strong", quality_score=0.9, speed_score=0.9, reliability_score=0.9, cost_score=0.9)
    router, *_ = _build({"p1": [weak], "p2": [strong]})
    decision = router.select(_req(), "balanced")
    assert decision.primary.provider_id == "p2"


def test_excludes_disabled_provider():
    m1 = make_model("p1", "m1")
    router, providers, *_ = _build({"p1": [m1]})
    providers.set_enabled("p1", False)
    with pytest.raises(NoAvailableModelError):
        router.select(_req())


def test_excludes_open_circuit():
    m1 = make_model("p1", "m1")
    m2 = make_model("p2", "m2")
    router, providers, circuits, _ = _build({"p1": [m1], "p2": [m2]})
    circuits.get("p1").force_open(cooldown_seconds=100)
    decision = router.select(_req())
    assert decision.primary.provider_id == "p2"


def test_excludes_offline_provider():
    m1 = make_model("p1", "m1")
    m2 = make_model("p2", "m2")
    router, providers, _, _ = _build({"p1": [m1], "p2": [m2]})
    providers.set_health("p1", ProviderHealth(status=ProviderStatus.OFFLINE))
    decision = router.select(_req())
    assert decision.primary.provider_id == "p2"


def test_all_unavailable_raises():
    m1 = make_model("p1", "m1")
    router, providers, *_ = _build({"p1": [m1]})
    providers.set_enabled("p1", False)
    with pytest.raises(NoAvailableModelError) as exc_info:
        router.select(_req())
    assert "No healthy" in str(exc_info.value)


def test_quota_critical_excludes_provider():
    m1 = make_model("p1", "m1")
    m2 = make_model("p2", "m2")
    router, providers, circuits, quota = _build({"p1": [m1], "p2": [m2]})
    quota.record_request("p1")
    quota.record_rate_limit("p1", retry_after=60.0)
    decision = router.select(_req())
    assert decision.primary.provider_id == "p2"


def test_local_preference_biases_toward_local():
    local = make_model("ollama", "local-model", quality_score=0.5, speed_score=0.5, reliability_score=0.5, cost_score=1.0)
    cloud = make_model("cloudp", "cloud-model", quality_score=0.5, speed_score=0.5, reliability_score=0.5, cost_score=0.5)
    router, *_ = _build({"ollama": [local], "cloudp": [cloud]}, local_ids={"ollama"})
    decision = router.select(_req(), "cheapest")
    assert decision.primary.provider_id == "ollama"


class _StubPolicyLearner:
    """Minimal stand-in for PolicyLearner exposing only the one method
    AdaptiveRouter.resolve_policy actually calls -- lets these tests assert
    the injection point in isolation from PolicyLearner's own learning
    logic (covered separately in tests/routing/test_policy_learner.py)."""

    def __init__(self, overrides: dict[str, PolicyWeights]):
        self._overrides = overrides

    def weights_for(self, policy_key: str, base: PolicyWeights) -> PolicyWeights:
        return self._overrides.get(policy_key, base)


def test_resolve_policy_uses_learned_override_when_present():
    override = PolicyWeights(quality=9.0, speed=9.0, reliability=9.0, cost=9.0, quota_risk=9.0, local_preference=9.0)
    router, *_ = _build({})
    router._policy_learner = _StubPolicyLearner({"quality": override})

    key, weights = router.resolve_policy("quality")

    assert key == "quality"
    assert weights is override


def test_resolve_policy_falls_back_to_base_weights_for_untouched_policy():
    override = PolicyWeights(quality=9.0)
    router, *_ = _build({})
    router._policy_learner = _StubPolicyLearner({"quality": override})

    key, weights = router.resolve_policy("balanced")

    assert key == "balanced"
    assert weights is POLICY_WEIGHTS["balanced"]


def test_resolve_policy_with_no_policy_learner_uses_base_weights():
    router, *_ = _build({})
    assert router._policy_learner is None  # existing construction path, unchanged

    key, weights = router.resolve_policy("cheapest")

    assert key == "cheapest"
    assert weights is POLICY_WEIGHTS["cheapest"]


def test_fallback_chain_ordered_by_score():
    weak = make_model("p1", "weak", quality_score=0.1, speed_score=0.1, reliability_score=0.5, cost_score=0.1)
    mid = make_model("p2", "mid", quality_score=0.5, speed_score=0.5, reliability_score=0.5, cost_score=0.5)
    strong = make_model("p3", "strong", quality_score=0.9, speed_score=0.9, reliability_score=0.9, cost_score=0.9)
    router, *_ = _build({"p1": [weak], "p2": [mid], "p3": [strong]})
    decision = router.select(_req())
    order = [decision.primary.provider_id] + [c.provider_id for c in decision.fallback_chain]
    assert order == ["p3", "p2", "p1"]
