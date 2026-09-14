"""Quota intelligence: track usage/429s per provider so routing can react
*before* hitting a hard limit, not just after a 429 (section 十二)."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum


class QuotaRisk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class _ProviderQuotaState:
    window_seconds: float = 60.0
    request_timestamps: deque = field(default_factory=deque)
    rate_limit_timestamps: deque = field(default_factory=deque)
    consecutive_rate_limits: int = 0
    tokens_total: int = 0
    requests_total: int = 0
    cooldown_until: float | None = None

    def _trim(self, dq: deque, now: float) -> None:
        while dq and (now - dq[0]) > self.window_seconds:
            dq.popleft()

    def record_request(self, now: float) -> None:
        self.request_timestamps.append(now)
        self.requests_total += 1
        self._trim(self.request_timestamps, now)

    def record_success(self) -> None:
        self.consecutive_rate_limits = 0

    def record_rate_limit(self, now: float, retry_after: float | None) -> None:
        self.rate_limit_timestamps.append(now)
        self.consecutive_rate_limits += 1
        self._trim(self.rate_limit_timestamps, now)
        if retry_after:
            self.cooldown_until = now + retry_after

    def record_tokens(self, tokens: int) -> None:
        self.tokens_total += tokens

    def risk(self, now: float) -> QuotaRisk:
        if self.cooldown_until and now < self.cooldown_until:
            return QuotaRisk.CRITICAL
        self._trim(self.request_timestamps, now)
        self._trim(self.rate_limit_timestamps, now)
        recent_requests = len(self.request_timestamps)
        recent_429s = len(self.rate_limit_timestamps)

        if self.consecutive_rate_limits >= 3:
            return QuotaRisk.CRITICAL
        if recent_requests == 0:
            return QuotaRisk.LOW
        ratio = recent_429s / recent_requests
        if ratio >= 0.5 or self.consecutive_rate_limits >= 2:
            return QuotaRisk.HIGH
        if ratio >= 0.15 or self.consecutive_rate_limits >= 1:
            return QuotaRisk.MEDIUM
        return QuotaRisk.LOW


class QuotaTracker:
    """One instance shared across the app; keyed by provider_id."""

    def __init__(self) -> None:
        self._state: dict[str, _ProviderQuotaState] = {}

    def _get(self, provider_id: str) -> _ProviderQuotaState:
        if provider_id not in self._state:
            self._state[provider_id] = _ProviderQuotaState()
        return self._state[provider_id]

    def record_request(self, provider_id: str) -> None:
        self._get(provider_id).record_request(time.time())

    def record_success(self, provider_id: str) -> None:
        self._get(provider_id).record_success()

    def record_rate_limit(self, provider_id: str, retry_after: float | None = None) -> None:
        self._get(provider_id).record_rate_limit(time.time(), retry_after)

    def record_tokens(self, provider_id: str, tokens: int) -> None:
        self._get(provider_id).record_tokens(tokens)

    def risk(self, provider_id: str) -> QuotaRisk:
        return self._get(provider_id).risk(time.time())

    def is_paused(self, provider_id: str) -> bool:
        return self.risk(provider_id) == QuotaRisk.CRITICAL

    def snapshot(self) -> dict:
        now = time.time()
        return {
            pid: {
                "requests_total": s.requests_total,
                "tokens_total": s.tokens_total,
                "recent_requests": len(s.request_timestamps),
                "recent_429s": len(s.rate_limit_timestamps),
                "consecutive_rate_limits": s.consecutive_rate_limits,
                "risk": s.risk(now).value,
                "cooldown_until": s.cooldown_until,
            }
            for pid, s in self._state.items()
        }


_tracker: QuotaTracker | None = None


def get_quota_tracker() -> QuotaTracker:
    global _tracker
    if _tracker is None:
        _tracker = QuotaTracker()
    return _tracker
