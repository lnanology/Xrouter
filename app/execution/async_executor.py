"""Executes a single provider attempt: concurrency admission -> bounded
timeout -> the actual provider call. Circuit breaker / retry / quota
bookkeeping happens one layer up in routing/fallback.py, which calls this
per attempt."""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

from app.reliability.timeout import with_timeout
from app.routing.scheduler import ConcurrencyLimiter

T = TypeVar("T")


async def execute_attempt(
    provider_id: str,
    limiter: ConcurrencyLimiter,
    timeout_seconds: float,
    call: Callable[[], Awaitable[T]],
) -> T:
    async with limiter.acquire(provider_id):
        return await with_timeout(call(), timeout_seconds, provider_id=provider_id)
