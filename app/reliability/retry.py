"""Bounded retry with exponential backoff + jitter. Only retries errors
explicitly marked retryable (see app.core.errors). Retry-After from the
provider always takes priority over computed backoff."""
from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from app.core.errors import ProviderError, ProviderRateLimitError

T = TypeVar("T")


@dataclass
class RetryConfig:
    max_attempts: int = 2  # additional attempts after the first try
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 8.0
    jitter_seconds: float = 0.25


async def retry_with_backoff(
    fn: Callable[[], Awaitable[T]],
    config: RetryConfig | None = None,
) -> T:
    cfg = config or RetryConfig()
    attempt = 0
    last_error: Exception | None = None

    while attempt <= cfg.max_attempts:
        try:
            return await fn()
        except ProviderError as e:
            last_error = e
            if not e.retryable or attempt == cfg.max_attempts:
                raise
            if isinstance(e, ProviderRateLimitError) and e.retry_after:
                delay = min(e.retry_after, cfg.max_delay_seconds)
            else:
                delay = min(cfg.base_delay_seconds * (2**attempt), cfg.max_delay_seconds)
            delay += random.uniform(0, cfg.jitter_seconds)
            await asyncio.sleep(delay)
            attempt += 1

    # Unreachable, but keeps type checkers happy.
    assert last_error is not None
    raise last_error
