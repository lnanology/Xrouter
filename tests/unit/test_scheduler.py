import asyncio

import pytest

from app.routing.scheduler import ConcurrencyLimiter


@pytest.mark.asyncio
async def test_provider_concurrency_is_bounded():
    limiter = ConcurrencyLimiter(global_limit=100)
    limiter.configure_provider("p1", 2)

    in_flight = 0
    max_in_flight = 0
    lock = asyncio.Lock()

    async def task():
        nonlocal in_flight, max_in_flight
        async with limiter.acquire("p1"):
            async with lock:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.02)
            async with lock:
                in_flight -= 1

    await asyncio.gather(*(task() for _ in range(10)))
    assert max_in_flight <= 2


@pytest.mark.asyncio
async def test_global_limit_applies_across_providers():
    limiter = ConcurrencyLimiter(global_limit=1)
    limiter.configure_provider("p1", 10)
    limiter.configure_provider("p2", 10)

    in_flight = 0
    max_in_flight = 0
    lock = asyncio.Lock()

    async def task(pid):
        nonlocal in_flight, max_in_flight
        async with limiter.acquire(pid):
            async with lock:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.01)
            async with lock:
                in_flight -= 1

    await asyncio.gather(*(task("p1") for _ in range(3)), *(task("p2") for _ in range(3)))
    assert max_in_flight == 1
