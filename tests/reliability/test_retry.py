import time

import pytest

from app.core.errors import ProviderAuthError, ProviderRateLimitError, ProviderTimeoutError
from app.reliability.retry import RetryConfig, retry_with_backoff


@pytest.mark.asyncio
async def test_retries_transient_errors_and_succeeds():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ProviderTimeoutError("timeout", provider_id="p1")
        return "ok"

    result = await retry_with_backoff(flaky, RetryConfig(max_attempts=3, base_delay_seconds=0.001, jitter_seconds=0.001))
    assert result == "ok"
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_does_not_retry_non_retryable():
    calls = {"n": 0}

    async def always_auth_error():
        calls["n"] += 1
        raise ProviderAuthError("bad key", provider_id="p1")

    with pytest.raises(ProviderAuthError):
        await retry_with_backoff(always_auth_error, RetryConfig(max_attempts=5))
    assert calls["n"] == 1  # never retried


@pytest.mark.asyncio
async def test_bounded_by_max_attempts():
    calls = {"n": 0}

    async def always_fails():
        calls["n"] += 1
        raise ProviderTimeoutError("timeout", provider_id="p1")

    with pytest.raises(ProviderTimeoutError):
        await retry_with_backoff(always_fails, RetryConfig(max_attempts=2, base_delay_seconds=0.001, jitter_seconds=0.001))
    assert calls["n"] == 3  # 1 initial + 2 retries, then raise


@pytest.mark.asyncio
async def test_honors_retry_after_over_backoff():
    calls = {"n": 0}
    start = time.time()

    async def rate_limited_then_ok():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ProviderRateLimitError("429", provider_id="p1", retry_after=0.05)
        return "ok"

    result = await retry_with_backoff(rate_limited_then_ok, RetryConfig(max_attempts=2, base_delay_seconds=5.0))
    elapsed = time.time() - start
    assert result == "ok"
    assert elapsed < 1.0  # used retry_after (0.05s), not the 5s base_delay
