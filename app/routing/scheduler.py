"""Concurrency admission control (section 十三). Bounds how many in-flight
requests are allowed per provider, per model, and globally, so XRouter never
hammers a provider past what it/the operator configured."""
from __future__ import annotations

import asyncio


class ConcurrencyLimiter:
    def __init__(self, global_limit: int = 64):
        self._global = asyncio.Semaphore(global_limit)
        self._provider: dict[str, asyncio.Semaphore] = {}
        self._provider_limits: dict[str, int] = {}

    def configure_provider(self, provider_id: str, limit: int) -> None:
        self._provider_limits[provider_id] = limit
        self._provider[provider_id] = asyncio.Semaphore(limit)

    def _provider_sem(self, provider_id: str) -> asyncio.Semaphore:
        if provider_id not in self._provider:
            self._provider[provider_id] = asyncio.Semaphore(4)
        return self._provider[provider_id]

    class _Lease:
        def __init__(self, sems: list[asyncio.Semaphore]):
            self._sems = sems

        async def __aenter__(self):
            for s in self._sems:
                await s.acquire()
            return self

        async def __aexit__(self, exc_type, exc, tb):
            for s in self._sems:
                s.release()

    def acquire(self, provider_id: str) -> "ConcurrencyLimiter._Lease":
        return self._Lease([self._global, self._provider_sem(provider_id)])
