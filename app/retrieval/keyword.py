"""The always-on, zero-extra-cost Retriever (app/retrieval/base.py):
keyword/substring search over app/storage/repositories/memory.py's
memory_entries, ordered by recency among matches. Deliberately not
"semantic" retrieval on its own -- and faking similarity search with e.g.
naive cosine-on-word-overlap and calling it RAG would be exactly the
"fake placeholder functionality" XRouter's own development rules forbid.
Keyword search is not a compromise dressed up as something fancier -- it
is a real, working retrieval step (retrieve -> augment the next call's
context -> generate), which is what actually makes this RAG rather than
nothing, and it stays the default because it costs nothing extra. A
genuine embedding-based Retriever now also exists
(app/retrieval/embedding.py's EmbeddingRetriever, real vector similarity
via a provider's embed() call) as an opt-in upgrade -- see its own
docstring and RetrievalConfig in app/core/config.py."""
from __future__ import annotations

from app.retrieval.base import RetrievedChunk
from app.storage.repositories.memory import MemoryRepository


class KeywordRetriever:
    def __init__(self, memory_repo: MemoryRepository):
        self._memory = memory_repo

    async def retrieve(self, scope: str, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        entries = await self._memory.search(scope, query, limit=top_k)
        return [RetrievedChunk(content=e.content, source="memory", score=1.0) for e in entries]
