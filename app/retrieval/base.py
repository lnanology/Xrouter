"""Retrieval interface (Phase 3: RAG groundwork). Swappable like every
other capability in XRouter (Provider, ExecutableTool): two real
implementations exist against this same Protocol --
app/retrieval/keyword.py's KeywordRetriever (substring search over
app/storage/repositories/memory.py, always available, zero extra cost)
and app/retrieval/embedding.py's EmbeddingRetriever (real embedding +
cosine-similarity search, opt-in via RetrievalConfig.enabled since it
spends a real provider call per recall/remember). app/agents/
orchestrator.py's `_build_retriever()` picks between them from config;
neither caller needed to change when the second implementation was
added. See app/retrieval/keyword.py's and app/retrieval/embedding.py's
own docstrings for why keyword retrieval ships as the always-on default
and embedding retrieval as the opt-in upgrade, rather than either one
faking the other."""
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
