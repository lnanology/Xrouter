"""Very simple linear predictor: given the current request rate and an
optional configured budget, estimate seconds-until-exhaustion. Used only as
an advisory metric surfaced on /admin/metrics — routing decisions rely on
the reactive QuotaTracker risk level, not this prediction."""
from __future__ import annotations

from app.quota.estimator import QuotaBudget


def seconds_until_exhaustion(budget: QuotaBudget, recent_requests_per_minute: float) -> float | None:
    if not budget.requests_per_minute or recent_requests_per_minute <= 0:
        return None
    remaining = max(budget.requests_per_minute - recent_requests_per_minute, 0)
    if remaining <= 0:
        return 0.0
    per_second_rate = recent_requests_per_minute / 60.0
    if per_second_rate <= 0:
        return None
    return remaining / per_second_rate
