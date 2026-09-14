"""Minimal in-memory async event bus. Other modules (metrics, logging,
telemetry writer) subscribe here instead of being called directly by the
router/engine, keeping those modules decoupled."""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger("xrouter.events")

Handler = Callable[[dict[str, Any]], Awaitable[None] | None]

# Canonical event type names (informational; not enforced).
EVENT_TYPES = [
    "request.received",
    "provider.selected",
    "request.started",
    "request.completed",
    "request.failed",
    "provider.unhealthy",
    "provider.recovered",
    "rate_limit.detected",
    "quota.warning",
    "cache.hit",
    "cache.miss",
    "fallback.triggered",
    "quality.failed",
    "routing.changed",
]


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = defaultdict(list)

    def subscribe(self, event_type: str, handler: Handler) -> None:
        self._handlers[event_type].append(handler)

    def emit(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        """Fire-and-forget: never lets a subscriber error break the caller."""
        payload = data or {}
        for handler in self._handlers.get(event_type, []):
            try:
                result = handler(payload)
                if asyncio.iscoroutine(result):
                    asyncio.ensure_future(result)
            except Exception:
                logger.exception("event handler failed for %s", event_type)


_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus
