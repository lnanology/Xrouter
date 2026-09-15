"""Retrieval interface (Phase 3: RAG groundwork). Swappable like every
other capability in XRouter (Provider, ExecutableTool): app/retrieval/
keyword.py is the one real implementation today, backed by
app/storage/repositories/memory.py's substring search. A future
embedding-backed implementation (real vector similarity, once XRouter
has an embedding provider and a vector store -- neither exists yet) can
implement this same Protocol and slot in without touching a single
caller. See app/retrieval/keyword.py's own docstring for why keyword
retrieval, not a faked-up "semantic" search, is what ships first."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    content: str
    source: str  # where this came from, e.g. "memory" -- for citation/debugging, not machine-parsed
    score: float = 0.0  # relevance signal; meaning depends on the retriever (e.g. keyword-match count), not comparable across implementations


@runtime_checkable
class Retriever(Protocol):
    async def retrieve(self, scope: str, query: str, top_k: int = 5) -> list[RetrievedChunk]: ...
