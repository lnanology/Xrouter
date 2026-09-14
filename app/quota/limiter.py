"""Translates quota risk into a routing weight multiplier and a hard pause
decision. Kept separate from tracker.py so the scoring formula can evolve
independently of how risk is measured."""
from __future__ import annotations

from app.quota.tracker import QuotaRisk

_WEIGHT_BY_RISK = {
    QuotaRisk.LOW: 1.0,
    QuotaRisk.MEDIUM: 0.6,
    QuotaRisk.HIGH: 0.25,
    QuotaRisk.CRITICAL: 0.0,
}


def weight_multiplier(risk: QuotaRisk) -> float:
    return _WEIGHT_BY_RISK[risk]


def should_pause(risk: QuotaRisk) -> bool:
    return risk == QuotaRisk.CRITICAL
