from app.quota.limiter import should_pause, weight_multiplier
from app.quota.tracker import QuotaRisk, QuotaTracker


def test_low_risk_when_no_activity():
    q = QuotaTracker()
    assert q.risk("p1") == QuotaRisk.LOW
    assert should_pause(q.risk("p1")) is False


def test_risk_escalates_with_consecutive_rate_limits():
    q = QuotaTracker()
    q.record_request("p1")
    q.record_rate_limit("p1")
    assert q.risk("p1") in (QuotaRisk.MEDIUM, QuotaRisk.HIGH)
    q.record_request("p1")
    q.record_rate_limit("p1")
    q.record_request("p1")
    q.record_rate_limit("p1")
    assert q.risk("p1") == QuotaRisk.CRITICAL
    assert should_pause(q.risk("p1")) is True


def test_cooldown_forces_critical():
    q = QuotaTracker()
    q.record_request("p1")
    q.record_rate_limit("p1", retry_after=60.0)
    assert q.risk("p1") == QuotaRisk.CRITICAL


def test_weight_multiplier_monotonic():
    assert weight_multiplier(QuotaRisk.LOW) > weight_multiplier(QuotaRisk.MEDIUM)
    assert weight_multiplier(QuotaRisk.MEDIUM) > weight_multiplier(QuotaRisk.HIGH)
    assert weight_multiplier(QuotaRisk.HIGH) > weight_multiplier(QuotaRisk.CRITICAL)
    assert weight_multiplier(QuotaRisk.CRITICAL) == 0.0


def test_success_resets_consecutive_rate_limits():
    q = QuotaTracker()
    q.record_request("p1")
    q.record_rate_limit("p1")
    q.record_success("p1")
    assert q._get("p1").consecutive_rate_limits == 0
