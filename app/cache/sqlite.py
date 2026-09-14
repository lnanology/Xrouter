"""L2 cache backed by the same SQLite database used for telemetry, so a
restart doesn't lose the cache. Uses aiosqlite for non-blocking access."""
from __future__ import annotations

import json
import time

import aiosqlite


class SqliteCache:
    def __init__(self, db_path: str):
        self._db_path = db_path

    async def get(self, key: str) -> dict | None:
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT value, created_at, ttl_seconds FROM cache_entries WHERE key = ?", (key,)
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            value, created_at, ttl_seconds = row
            if (time.time() - created_at) > ttl_seconds:
                await db.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                await db.commit()
                return None
            return json.loads(value)

    async def set(self, key: str, value: dict, ttl_seconds: float) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO cache_entries (key, value, created_at, ttl_seconds) VALUES (?, ?, ?, ?)",
                (key, json.dumps(value), time.time(), ttl_seconds),
            )
            await db.commit()
