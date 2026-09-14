"""Bounded timeout wrapper. Every outbound provider call must go through
this so a hung provider can never hang a request indefinitely."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

from app.core.errors import ProviderTimeoutError

T = TypeVar("T")


async def with_timeout(awaitable: Awaitable[T], seconds: float, *, provider_id: str) -> T:
    try:
        return await asyncio.wait_for(awaitable, timeout=seconds)
    except asyncio.TimeoutError as e:
        raise ProviderTimeoutError(
            f"Provider '{provider_id}' timed out after {seconds}s", provider_id=provider_id
        ) from e
