from __future__ import annotations

import time

from app.contracts.model import ModelInfo
from app.observability.logging import get_logger
from app.storage.database import Database

logger = get_logger("repo.model")


class ModelRepository:
    def __init__(self, db: Database):
        self._db = db

    async def upsert(self, model: ModelInfo) -> None:
        try:
            async with self._db.connect() as conn:
                await conn.execute(
                    "INSERT INTO models (id, provider_id, name, status, quality_score, speed_score,"
                    " reliability_score, cost_score, updated_at) VALUES (?,?,?,?,?,?,?,?,?)"
                    " ON CONFLICT(id) DO UPDATE SET status=excluded.status, quality_score=excluded.quality_score,"
                    " speed_score=excluded.speed_score, reliability_score=excluded.reliability_score,"
                    " cost_score=excluded.cost_score, updated_at=excluded.updated_at",
                    (
                        model.public_id(),
                        model.provider_id,
                        model.name,
                        model.status.value,
                        model.quality_score,
                        model.speed_score,
                        model.reliability_score,
                        model.cost_score,
                        time.time(),
                    ),
                )
                await conn.commit()
        except Exception as e:
            logger.warning("model upsert failed: %s", e)

    async def upsert_many(self, models: list[ModelInfo]) -> None:
        for m in models:
            await self.upsert(m)
