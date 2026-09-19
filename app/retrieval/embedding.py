"""The opt-in, embedding-backed Retriever (app/retrieval/base.py) --
real vector similarity, not a heuristic dressed up as one. Off by
default (RetrievalConfig.enabled in app/core/config.py): it spends a
real provider embed() call on every recall *and* every remember at
Orchestrator tier >= 2, on top of what app/retrieval/keyword.py's
KeywordRetriever already does for free.

Similarity is computed in pure Python (_cosine_similarity below) rather
than with numpy -- a deliberate choice, not an oversight: XRouter has no
numerical-computation dependency today, and per-scope memory volume is
small enough (app/storage/repositories/memory.py's own search()/
embedded() scan caps) that a hand-written dot-product loop is genuinely
fast enough. Adding numpy purely to vectorize a loop over a few hundred
short vectors would be exactly the unnecessary infrastructure XRouter's
own development rules warn against.

Both entry points here fail open, never raising into their caller:
retrieve() returns [] and embed_for_memory() returns None on a missing
provider, an unconfigured/unsupported adapter, or any ProviderError.
EmbeddingRetriever deliberately does NOT fall back to KeywordRetriever
internally on failure -- an empty result is the same "nothing found"
shape a genuinely empty scope already produces, which keeps the two
retrievers simple to reason about independently. app/agents/
orchestrator.py's _build_retriever() is what decides which one runs, from
config, once per call."""
from __future__ import annotations

import math

from app.contracts.memory import MemoryEntry
from app.core.errors import ProviderError
from app.core.registry import ProviderRegistry
from app.retrieval.base import RetrievedChunk
from app.storage.repositories.memory import MemoryRepository


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure Python -- no new dependency. Same math a vectorized version
    would do, just as a plain loop. Returns 0.0 (never raises) for any
    degenerate input: empty vectors, mismatched dimensions, or a
    zero-magnitude vector."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


async def embed_for_memory(providers: ProviderRegistry, provider_id: str, model: str, text: str) -> list[float] | None:
    """Write-path helper for app/agents/orchestrator.py's _remember():
    embeds one piece of text for storage. Fails open to None on a missing
    provider or any ProviderError -- the memory entry is still saved (via
    MemoryRepository.save()'s own default embedding=None), just invisible
    to EmbeddingRetriever's cosine scan until/unless it's re-embedded."""
    provider = providers.get(provider_id)
    if provider is None:
        return None
    try:
        vectors = await provider.embed(model, [text])
    except ProviderError:
        return None
    return vectors[0] if vectors else None


class EmbeddingRetriever:
    """Retriever Protocol implementation #2 (app/retrieval/base.py) --
    embeds the query, then ranks app/storage/repositories/memory.py's
    embedded() candidate pool by cosine similarity. Unlike
    KeywordRetriever's arbitrary fixed score, `RetrievedChunk.score` here
    is a genuine, comparable-within-this-retriever similarity value."""

    def __init__(
        self,
        memory_repo: MemoryRepository,
        providers: ProviderRegistry,
        provider_id: str,
        model: str,
        max_candidates: int = 200,
    ):
        self._memory = memory_repo
        self._providers = providers
        self._provider_id = provider_id
        self._model = model
        self._max_candidates = max_candidates

    async def retrieve(self, scope: str, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        provider = self._providers.get(self._provider_id)
        if provider is None:
            return []
        try:
            vectors = await provider.embed(self._model, [query])
        except ProviderError:
            return []
        if not vectors:
            return []
        query_vec = vectors[0]

        candidates: list[MemoryEntry] = await self._memory.embedded(scope, limit=self._max_candidates)
        scored = sorted(
            ((c, _cosine_similarity(query_vec, c.embedding)) for c in candidates if c.embedding),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [RetrievedChunk(content=c.content, source="memory", score=score) for c, score in scored[:top_k]]
