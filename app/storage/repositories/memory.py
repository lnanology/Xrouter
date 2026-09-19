"""Backs memory_entries (Phase 3: Memory/RAG groundwork). `search()` is a
real, working substring match over stored content -- app/retrieval/
keyword.py's KeywordRetriever, always available, never a fake stand-in for
semantic similarity dressed up as one. `save()` can also optionally carry
a real embedding vector (`embedding_json`, nullable -- most rows have
none) when app/retrieval/embedding.py's opt-in EmbeddingRetriever path is
enabled; `embedded()` is that retriever's own candidate-pool query. See
app/retrieval/'s module docstring for the two Retriever implementations
this repository backs."""
from __future__ import annotations

import json
import time

from app.contracts.memory import MemoryEntry
from app.observability.logging import get_logger
from app.storage.database import Database
from app.utils.ids import new_id

logger = get_logger("repo.memory")

# Common function words carry no topical signal and would otherwise match
# almost every stored entry by pure chance (e.g. "the", "about") -- excluding
# them keeps search() "permissive but not noisy": still no real relevance
# ranking, just a saner definition of "meaningful word" than length alone.
_STOPWORDS = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "her",
    "was", "one", "our", "out", "day", "get", "has", "him", "his", "how",
    "man", "new", "now", "old", "see", "two", "way", "who", "boy", "did",
    "its", "let", "put", "say", "she", "too", "use", "about", "tell", "me",
})


class MemoryRepository:
    def __init__(self, db: Database):
        self._db = db

    async def save(
        self, scope: str, content: str, tags: list[str] | None = None, embedding: list[float] | None = None,
    ) -> MemoryEntry:
        entry = MemoryEntry(
            id=new_id("mem"), scope=scope, content=content, tags=tags or [], created_at=time.time(), embedding=embedding,
        )
        try:
            async with self._db.connect() as conn:
                await conn.execute(
                    "INSERT INTO memory_entries (id, scope, content, tags_json, created_at, embedding_json) VALUES (?,?,?,?,?,?)",
                    (
                        entry.id, entry.scope, entry.content, json.dumps(entry.tags), entry.created_at,
                        json.dumps(entry.embedding) if entry.embedding is not None else None,
                    ),
                )
                await conn.commit()
        except Exception as e:
            logger.warning("memory save failed: %s", e)
        return entry

    async def search(self, scope: str, query: str, limit: int = 5) -> list[MemoryEntry]:
        """Substring match (case-insensitive) over `content`, newest
        first. `query` is split into words and an entry must contain at
        least one of them -- deliberately permissive (recall, not
        precision) since a caller-side LLM step reads the results and can
        judge relevance itself, the same way a human skims search hits."""
        words = [w.lower() for w in query.split() if len(w) > 2 and w.lower() not in _STOPWORDS]
        if not words:
            return await self.recent(scope, limit)
        try:
            async with self._db.connect() as conn:
                cursor = await conn.execute(
                    "SELECT id, scope, content, tags_json, created_at, embedding_json FROM memory_entries"
                    " WHERE scope = ? ORDER BY created_at DESC LIMIT 200",
                    (scope,),
                )
                rows = await cursor.fetchall()
        except Exception as e:
            logger.warning("memory search failed: %s", e)
            return []

        matches = []
        for r in rows:
            content_lower = r[2].lower()
            if any(w in content_lower for w in words):
                matches.append(_row_to_entry(r))
        return matches[:limit]

    async def recent(self, scope: str, limit: int = 5) -> list[MemoryEntry]:
        try:
            async with self._db.connect() as conn:
                cursor = await conn.execute(
                    "SELECT id, scope, content, tags_json, created_at, embedding_json FROM memory_entries"
                    " WHERE scope = ? ORDER BY created_at DESC LIMIT ?",
                    (scope, limit),
                )
                rows = await cursor.fetchall()
                return [_row_to_entry(r) for r in rows]
        except Exception as e:
            logger.warning("memory recent query failed: %s", e)
            return []

    async def embedded(self, scope: str, limit: int = 200) -> list[MemoryEntry]:
        """The candidate pool app/retrieval/embedding.py's EmbeddingRetriever
        scans for a query -- every entry in `scope` that actually has an
        embedding, newest first, capped the same way search()'s own scan
        is capped. Entries saved before retrieval was enabled (or whose
        embed() call itself failed open) simply have no embedding and are
        invisible here -- not an error, just nothing to rank them by."""
        try:
            async with self._db.connect() as conn:
                cursor = await conn.execute(
                    "SELECT id, scope, content, tags_json, created_at, embedding_json FROM memory_entries"
                    " WHERE scope = ? AND embedding_json IS NOT NULL ORDER BY created_at DESC LIMIT ?",
                    (scope, limit),
                )
                rows = await cursor.fetchall()
                return [_row_to_entry(r) for r in rows]
        except Exception as e:
            logger.warning("memory embedded query failed: %s", e)
            return []


def _row_to_entry(r) -> MemoryEntry:
    embedding = json.loads(r[5]) if len(r) > 5 and r[5] is not None else None
    return MemoryEntry(id=r[0], scope=r[1], content=r[2], tags=json.loads(r[3]), created_at=r[4], embedding=embedding)
