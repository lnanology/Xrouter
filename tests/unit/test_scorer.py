from app.contracts.policy import PolicyWeights
from app.contracts.provider import ProviderHealth, ProviderStatus
from app.quota.tracker import QuotaRisk
from app.reliability.circuit_breaker import CircuitState
from app.routing.scorer import ScoreInput, score_candidate
from tests.helpers import make_model


def test_excludes_when_streaming_required_but_unsupported():
    model = make_model("p1", supports_streaming=False)
    inp = ScoreInput(
        model=model, provider_health=ProviderHealth(status=ProviderStatus.HEALTHY, success_rate=1.0),
        circuit_state=CircuitState.CLOSED, quota_risk=QuotaRisk.LOW, requires_streaming=True,
    )
    assert score_candidate(inp, PolicyWeights(), is_local=False) is None


def test_excludes_when_circuit_open():
    model = make_model("p1")
    inp = ScoreInput(
        model=model, provider_health=ProviderHealth(status=ProviderStatus.HEALTHY, success_rate=1.0),
        circuit_state=CircuitState.OPEN, quota_risk=QuotaRisk.LOW,
    )
    assert score_candidate(inp, PolicyWeights(), is_local=False) is None


def test_excludes_when_quota_critical():
    model = make_model("p1")
    inp = ScoreInput(
        model=model, provider_health=ProviderHealth(status=ProviderStatus.HEALTHY, success_rate=1.0),
        circuit_state=CircuitState.CLOSED, quota_risk=QuotaRisk.CRITICAL,
    )
    assert score_candidate(inp, PolicyWeights(), is_local=False) is None


def test_higher_quality_scores_higher_under_quality_policy():
    low = make_model("p1", quality_score=0.2)
    high = make_model("p2", quality_score=0.9)
    health = ProviderHealth(status=ProviderStatus.HEALTHY, success_rate=1.0)
    weights = PolicyWeights(quality=3.0)
    s_low = score_candidate(ScoreInput(low, health, CircuitState.CLOSED, QuotaRisk.LOW), weights, False)
    s_high = score_candidate(ScoreInput(high, health, CircuitState.CLOSED, QuotaRisk.LOW), weights, False)
    assert s_high > s_low


def test_local_preference_multiplies_score():
    model = make_model("p1")
    health = ProviderHealth(status=ProviderStatus.HEALTHY, success_rate=1.0)
    weights = PolicyWeights(local_preference=1.0)
    s_local = score_candidate(ScoreInput(model, health, CircuitState.CLOSED, QuotaRisk.LOW), weights, True)
    s_cloud = score_candidate(ScoreInput(model, health, CircuitState.CLOSED, QuotaRisk.LOW), weights, False)
    assert s_local > s_cloud
