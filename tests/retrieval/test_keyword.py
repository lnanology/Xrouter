import pytest

from app.retrieval.base import RetrievedChunk
from app.retrieval.keyword import KeywordRetriever
from app.storage.database import Database
from app.storage.repositories.memory import MemoryRepository


@pytest.fixture
async def retriever(tmp_path):
    db = Database(str(tmp_path / "test.sqlite3"))
    await db.init()
    memory = MemoryRepository(db)
    return KeywordRetriever(memory), memory


@pytest.mark.asyncio
async def test_retrieve_wraps_matching_memory_entries_as_chunks(retriever):
    kw, memory = retriever
    await memory.save("global", "XRouter is a provider-agnostic AI gateway")
    chunks = await kw.retrieve("global", "tell me about the gateway", top_k=5)
    assert len(chunks) == 1
    assert isinstance(chunks[0], RetrievedChunk)
    assert chunks[0].content == "XRouter is a provider-agnostic AI gateway"
    assert chunks[0].source == "memory"


@pytest.mark.asyncio
async def test_retrieve_respects_top_k(retriever):
    kw, memory = retriever
    for i in range(5):
        await memory.save("global", f"fact number {i} about widgets")
    chunks = await kw.retrieve("global", "widgets", top_k=2)
    assert len(chunks) == 2


@pytest.mark.asyncio
async def test_retrieve_returns_empty_list_when_nothing_matches(retriever):
    kw, memory = retriever
    await memory.save("global", "completely unrelated content")
    chunks = await kw.retrieve("global", "quantum mechanics", top_k=5)
    assert chunks == []


@pytest.mark.asyncio
async def test_retrieve_is_scoped(retriever):
    kw, memory = retriever
    await memory.save("scope-a", "the secret ingredient is basil")
    chunks = await kw.retrieve("scope-b", "secret ingredient", top_k=5)
    assert chunks == []
