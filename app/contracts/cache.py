"""Cache contracts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CacheEntry:
    key: str
    value: Any
    created_at: float
    ttl_seconds: float
    hits: int = 0

    def is_expired(self, now: float) -> bool:
        return (now - self.created_at) > self.ttl_seconds
