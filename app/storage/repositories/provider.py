from __future__ import annotations

import time

from app.observability.logging import get_logger
from app.storage.database import Database

logger = get_logger("repo.provider")


class ProviderRepository:
    def __init__(self, db: Database):
        self._db = db

    async def upsert(self, provider_id: str, name: str, type_: str, enabled: bool) -> None:
        try:
            async with self._db.connect() as conn:
                await conn.execute(
                    "INSERT INTO providers (id, name, type, enabled, updated_at) VALUES (?,?,?,?,?)"
                    " ON CONFLICT(id) DO UPDATE SET name=excluded.name, type=excluded.type, "
                    "enabled=excluded.enabled, updated_at=excluded.updated_at",
                    (provider_id, name, type_, int(enabled), time.time()),
                )
                await conn.commit()
        except Exception as e:
            logger.warning("provider upsert failed: %s", e)

    async def record_health(self, provider_id: str, status: str, latency_ms: float | None, success_rate: float, last_error: str | None) -> None:
        try:
            async with self._db.connect() as conn:
                await conn.execute(
                    "INSERT INTO provider_health (provider_id, status, latency_ms, success_rate, last_error, checked_at)"
                    " VALUES (?,?,?,?,?,?)",
                    (provider_id, status, latency_ms, success_rate, last_error, time.time()),
                )
                await conn.commit()
        except Exception as e:
            logger.warning("record_health failed: %s", e)

    async def recent_health(self, provider_id: str, limit: int = 20) -> list[dict]:
        try:
            async with self._db.connect() as conn:
                cursor = await conn.execute(
                    "SELECT status, latency_ms, success_rate, last_error, checked_at FROM provider_health"
                    " WHERE provider_id = ? ORDER BY checked_at DESC LIMIT ?",
                    (provider_id, limit),
                )
                rows = await cursor.fetchall()
                return [
                    {"status": r[0], "latency_ms": r[1], "success_rate": r[2], "last_error": r[3], "checked_at": r[4]}
                    for r in rows
                ]
        except Exception as e:
            logger.warning("recent_health failed: %s", e)
            return []
