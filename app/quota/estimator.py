"""Optional static quota budgets from config (requests/min, tokens/day).
When a provider has no configured budget, estimation is skipped and only
the reactive tracker (429-based) drives risk — this is intentional: XRouter
never assumes a free provider's limit, per rule 十八/三十七."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QuotaBudget:
    requests_per_minute: int | None = None
    tokens_per_day: int | None = None


def budget_from_extra(extra: dict) -> QuotaBudget:
    return QuotaBudget(
        requests_per_minute=extra.get("quota_requests_per_minute"),
        tokens_per_day=extra.get("quota_tokens_per_day"),
    )


def usage_fraction(budget: QuotaBudget, recent_requests_per_minute: float, tokens_today: int) -> float | None:
    """Returns the worst-case fraction (0..1+) of configured budget consumed,
    or None if no budget is configured for this provider."""
    fractions = []
    if budget.requests_per_minute:
        fractions.append(recent_requests_per_minute / budget.requests_per_minute)
    if budget.tokens_per_day:
        fractions.append(tokens_today / budget.tokens_per_day)
    if not fractions:
        return None
    return max(fractions)
