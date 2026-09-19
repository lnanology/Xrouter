"""Memory contracts (Phase 3: Dynamic Agent Team groundwork). A memory
entry is one durable fact/summary XRouter chose to remember -- written by
app/agents/orchestrator.py after a tier>=2 run, read back by
app/retrieval/ before the next run on a matching scope. See
app/storage/repositories/memory.py for how entries are actually stored
and searched."""
from __future__ import annotations

from pydantic import BaseModel, Field


class MemoryEntry(BaseModel):
    id: str
    # Groups entries so recall never leaks across unrelated contexts --
    # "global" for anything not otherwise scoped, or a caller-supplied
    # scope (e.g. a session/user id) for anything narrower. Never
    # inferred automatically from message content.
    scope: str = "global"
    content: str
    tags: list[str] = Field(default_factory=list)
    created_at: float
    # Set only when Retrieval Config's embedding-backed path is enabled
    # and app/retrieval/embedding.py successfully embedded this entry at
    # save time -- None for every entry saved with retrieval off (the
    # default), and for one whose embed() call failed open. See
    # app/retrieval/embedding.py's EmbeddingRetriever, the only reader.
    embedding: list[float] | None = None
