from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.observability.logging import get_logger
from app.storage.database import Database

logger = get_logger("repo.request")


@dataclass
class RequestRecord:
    request_id: str
    timestamp: float = field(default_factory=time.time)
    task_type: str = "chat"
    complexity: int = 0
    routing_policy: str = "balanced"
    status: str = "pending"
    latency_ms: float | None = None
    ttft_ms: float | None = None
    tokens_prompt: int = 0
    tokens_completion: int = 0
    retry_count: int = 0
    fallback_count: int = 0
    cache_hit: bool = False
    error: str | None = None


class RequestRepository:
    def __init__(self, db: Database):
        self._db = db

    async def save(self, r: RequestRecord) -> None:
        try:
            async with self._db.connect() as conn:
                await conn.execute(
                    "INSERT INTO requests (request_id, timestamp, task_type, complexity, routing_policy, status,"
                    " latency_ms, ttft_ms, tokens_prompt, tokens_completion, retry_count, fallback_count, cache_hit, error)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
                    " ON CONFLICT(request_id) DO UPDATE SET status=excluded.status, latency_ms=excluded.latency_ms,"
                    " ttft_ms=excluded.ttft_ms, tokens_prompt=excluded.tokens_prompt,"
                    " tokens_completion=excluded.tokens_completion, retry_count=excluded.retry_count,"
                    " fallback_count=excluded.fallback_count, cache_hit=excluded.cache_hit, error=excluded.error",
                    (
                        r.request_id, r.timestamp, r.task_type, r.complexity, r.routing_policy, r.status,
                        r.latency_ms, r.ttft_ms, r.tokens_prompt, r.tokens_completion, r.retry_count,
                        r.fallback_count, int(r.cache_hit), r.error,
                    ),
                )
                await conn.commit()
        except Exception as e:
            logger.warning("request save failed: %s", e)

    async def record_attempt(
        self, request_id: str, attempt_index: int, provider_id: str, model_id: str, status: str,
        latency_ms: float | None, error: str | None,
    ) -> None:
        try:
            async with self._db.connect() as conn:
                await conn.execute(
                    "INSERT INTO request_attempts (request_id, attempt_index, provider_id, model_id, status,"
                    " latency_ms, error, timestamp) VALUES (?,?,?,?,?,?,?,?)",
                    (request_id, attempt_index, provider_id, model_id, status, latency_ms, error, time.time()),
                )
                await conn.commit()
        except Exception as e:
            logger.warning("record_attempt failed: %s", e)

    async def recent(self, limit: int = 50) -> list[dict]:
        try:
            async with self._db.connect() as conn:
                cursor = await conn.execute(
                    "SELECT request_id, timestamp, status, latency_ms, routing_policy, cache_hit, fallback_count"
                    " FROM requests ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                )
                rows = await cursor.fetchall()
                return [
                    {
                        "request_id": r[0], "timestamp": r[1], "status": r[2], "latency_ms": r[3],
                        "routing_policy": r[4], "cache_hit": bool(r[5]), "fallback_count": r[6],
                    }
                    for r in rows
                ]
        except Exception as e:
            logger.warning("recent requests query failed: %s", e)
            return []
