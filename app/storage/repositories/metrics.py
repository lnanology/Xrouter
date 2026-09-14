from __future__ import annotations

import json
import time

from app.observability.logging import get_logger
from app.storage.database import Database

logger = get_logger("repo.metrics")


class MetricsRepository:
    """Backs routing_metrics, quota_usage, benchmarks, and events tables."""

    def __init__(self, db: Database):
        self._db = db

    async def record_routing_metrics(self, metrics: dict) -> None:
        try:
            async with self._db.connect() as conn:
                await conn.execute(
                    "INSERT INTO routing_metrics (recorded_at, metrics_json) VALUES (?,?)",
                    (time.time(), json.dumps(metrics)),
                )
                await conn.commit()
        except Exception as e:
            logger.warning("record_routing_metrics failed: %s", e)

    async def record_quota_usage(self, provider_id: str, requests_total: int, tokens_total: int, risk: str) -> None:
        try:
            async with self._db.connect() as conn:
                await conn.execute(
                    "INSERT INTO quota_usage (provider_id, requests_total, tokens_total, risk, recorded_at)"
                    " VALUES (?,?,?,?,?)",
                    (provider_id, requests_total, tokens_total, risk, time.time()),
                )
                await conn.commit()
        except Exception as e:
            logger.warning("record_quota_usage failed: %s", e)

    async def record_benchmark(
        self, provider_id: str, model_id: str, ttft_ms: float | None, total_latency_ms: float | None,
        tokens_per_sec: float | None, success: bool,
    ) -> None:
        try:
            async with self._db.connect() as conn:
                await conn.execute(
                    "INSERT INTO benchmarks (provider_id, model_id, ttft_ms, total_latency_ms, tokens_per_sec,"
                    " success, recorded_at) VALUES (?,?,?,?,?,?,?)",
                    (provider_id, model_id, ttft_ms, total_latency_ms, tokens_per_sec, int(success), time.time()),
                )
                await conn.commit()
        except Exception as e:
            logger.warning("record_benchmark failed: %s", e)

    async def record_event(self, event_type: str, payload: dict) -> None:
        try:
            async with self._db.connect() as conn:
                await conn.execute(
                    "INSERT INTO events (event_type, payload_json, recorded_at) VALUES (?,?,?)",
                    (event_type, json.dumps(payload, default=str), time.time()),
                )
                await conn.commit()
        except Exception as e:
            logger.warning("record_event failed: %s", e)

    async def recent_benchmarks(self, limit: int = 50) -> list[dict]:
        try:
            async with self._db.connect() as conn:
                cursor = await conn.execute(
                    "SELECT provider_id, model_id, ttft_ms, total_latency_ms, tokens_per_sec, success, recorded_at"
                    " FROM benchmarks ORDER BY recorded_at DESC LIMIT ?",
                    (limit,),
                )
                rows = await cursor.fetchall()
                return [
                    {
                        "provider_id": r[0], "model_id": r[1], "ttft_ms": r[2], "total_latency_ms": r[3],
                        "tokens_per_sec": r[4], "success": bool(r[5]), "recorded_at": r[6],
                    }
                    for r in rows
                ]
        except Exception as e:
            logger.warning("recent_benchmarks failed: %s", e)
            return []
