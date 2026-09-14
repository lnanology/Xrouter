"""Fronts L1 (memory) + L2 (sqlite) caches. Decides cacheability: streaming
requests are never cached; requests whose prompt text matches a configured
"volatile" keyword (news/price/weather/...) are never cached, since a
correct-looking stale answer is worse than a cache miss (section 十六)."""
from __future__ import annotations

from app.cache.memory import MemoryCache
from app.cache.sqlite import SqliteCache
from app.contracts.request import ChatCompletionRequest
from app.core.config import CacheConfig
from app.utils.json import stable_hash


class CacheManager:
    def __init__(self, config: CacheConfig, db_path: str):
        self.config = config
        self.l1 = MemoryCache(max_entries=config.l1_max_entries)
        self.l2 = SqliteCache(db_path)

    def is_cacheable(self, request: ChatCompletionRequest) -> bool:
        if not self.config.enabled:
            return False
        if request.stream:
            return False
        if request.x_cache is False:
            return False
        text = " ".join(
            (m.content if isinstance(m.content, str) else "") for m in request.messages
        ).lower()
        return not any(kw in text for kw in self.config.volatile_keywords)

    def key_for(self, request: ChatCompletionRequest) -> str:
        payload = {
            "model": request.model,
            "messages": [m.model_dump(exclude_none=True) for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        return stable_hash(payload)

    async def get(self, key: str) -> dict | None:
        entry = self.l1.get(key)
        if entry is not None:
            return entry.value
        value = await self.l2.get(key)
        if value is not None:
            self.l1.set(key, value, self.config.default_ttl_seconds)
        return value

    async def set(self, key: str, value: dict) -> None:
        self.l1.set(key, value, self.config.default_ttl_seconds)
        await self.l2.set(key, value, self.config.default_ttl_seconds)
