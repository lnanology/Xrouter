"""L1 in-memory cache. Simple bounded LRU-ish dict keyed by request hash."""
from __future__ import annotations

import time
from collections import OrderedDict

from app.contracts.cache import CacheEntry


class MemoryCache:
    def __init__(self, max_entries: int = 512):
        self._max_entries = max_entries
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()

    def get(self, key: str) -> CacheEntry | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        if entry.is_expired(time.time()):
            del self._store[key]
            return None
        entry.hits += 1
        self._store.move_to_end(key)
        return entry

    def set(self, key: str, value, ttl_seconds: float) -> None:
        self._store[key] = CacheEntry(key=key, value=value, created_at=time.time(), ttl_seconds=ttl_seconds)
        self._store.move_to_end(key)
        while len(self._store) > self._max_entries:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()

    def size(self) -> int:
        return len(self._store)
