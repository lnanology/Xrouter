import pytest

from app.contracts.provider import ProviderConfig
from app.core.registry import ProviderRegistry
from app.retrieval.base import RetrievedChunk
from app.retrieval.embedding import EmbeddingRetriever, _cosine_similarity, embed_for_memory
from app.storage.database import Database
from app.storage.repositories.memory import MemoryRepository
from tests.helpers import FakeProvider, make_model


def _provider(embed_vectors=None, behavior="success"):
    cfg = ProviderConfig(id="ollama", name="ollama", type="ollama")
    return FakeProvider(cfg, [make_model("ollama")], embed_vectors=embed_vectors, behavior=behavior)


def _registry(provider) -> ProviderRegistry:
    reg = ProviderRegistry()
    reg.register(provider, provider.config)
    return reg


@pytest.fixture
async def memory(tmp_path):
    db = Database(str(tmp_path / "test.sqlite3"))
    await db.init()
    return MemoryRepository(db)


# --- _cosine_similarity -----------------------------------------------------

def test_cosine_similarity_identical_vectors_is_one():
    assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors_is_negative_one():
    assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_similarity_handles_degenerate_input_without_raising():
    assert _cosine_similarity([], []) == 0.0
    assert _cosine_similarity([1.0], [1.0, 2.0]) == 0.0
    assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


# --- embed_for_memory --------------------------------------------------------

@pytest.mark.asyncio
async def test_embed_for_memory_returns_the_scripted_vector():
    provider = _provider(embed_vectors={"hello": [1.0, 2.0]})
    result = await embed_for_memory(_registry(provider), "ollama", "test-model", "hello")
    assert result == [1.0, 2.0]


@pytest.mark.asyncio
async def test_embed_for_memory_fails_open_on_missing_provider():
    result = await embed_for_memory(ProviderRegistry(), "ollama", "test-model", "hello")
    assert result is None


@pytest.mark.asyncio
async def test_embed_for_memory_fails_open_when_provider_does_not_support_embeddings():
    provider = _provider(embed_vectors=None)
    result = await embed_for_memory(_registry(provider), "ollama", "test-model", "hello")
    assert result is None


@pytest.mark.asyncio
async def test_embed_for_memory_fails_open_on_provider_error():
    provider = _provider(embed_vectors={"hello": [1.0]}, behavior="timeout")
    result = await embed_for_memory(_registry(provider), "ollama", "test-model", "hello")
    assert result is None


# --- EmbeddingRetriever.retrieve ---------------------------------------------

@pytest.mark.asyncio
async def test_retrieve_ranks_by_cosine_similarity(memory):
    provider = _provider(embed_vectors={"about the gateway": [1.0, 0.0]})
    await memory.save("global", "close match", embedding=[1.0, 0.0])
    await memory.save("global", "unrelated", embedding=[0.0, 1.0])
    retriever = EmbeddingRetriever(memory, _registry(provider), "ollama", "test-model")

    chunks = await retriever.retrieve("global", "about the gateway", top_k=5)

    assert [c.content for c in chunks] == ["close match", "unrelated"]
    assert isinstance(chunks[0], RetrievedChunk)
    assert chunks[0].score > chunks[1].score
    assert chunks[0].score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_retrieve_respects_top_k(memory):
    provider = _provider(embed_vectors={"q": [1.0, 0.0]})
    for i in range(5):
        await memory.save("global", f"entry {i}", embedding=[1.0, 0.0])
    retriever = EmbeddingRetriever(memory, _registry(provider), "ollama", "test-model")

    chunks = await retriever.retrieve("global", "q", top_k=2)
    assert len(chunks) == 2


@pytest.mark.asyncio
async def test_retrieve_is_scoped(memory):
    provider = _provider(embed_vectors={"q": [1.0, 0.0]})
    await memory.save("scope-a", "a-entry", embedding=[1.0, 0.0])
    retriever = EmbeddingRetriever(memory, _registry(provider), "ollama", "test-model")

    chunks = await retriever.retrieve("scope-b", "q", top_k=5)
    assert chunks == []


@pytest.mark.asyncio
async def test_retrieve_ignores_entries_with_no_embedding(memory):
    provider = _provider(embed_vectors={"q": [1.0, 0.0]})
    await memory.save("global", "no embedding here")  # embedding=None, the default
    retriever = EmbeddingRetriever(memory, _registry(provider), "ollama", "test-model")

    chunks = await retriever.retrieve("global", "q", top_k=5)
    assert chunks == []


@pytest.mark.asyncio
async def test_retrieve_returns_empty_list_when_provider_missing(memory):
    await memory.save("global", "some entry", embedding=[1.0, 0.0])
    retriever = EmbeddingRetriever(memory, ProviderRegistry(), "ollama", "test-model")

    chunks = await retriever.retrieve("global", "q", top_k=5)
    assert chunks == []


@pytest.mark.asyncio
async def test_retrieve_returns_empty_list_when_provider_does_not_support_embeddings(memory):
    provider = _provider(embed_vectors=None)
    await memory.save("global", "some entry", embedding=[1.0, 0.0])
    retriever = EmbeddingRetriever(memory, _registry(provider), "ollama", "test-model")

    chunks = await retriever.retrieve("global", "q", top_k=5)
    assert chunks == []


@pytest.mark.asyncio
async def test_retrieve_returns_empty_list_on_provider_error(memory):
    provider = _provider(embed_vectors={"q": [1.0, 0.0]}, behavior="server_error")
    await memory.save("global", "some entry", embedding=[1.0, 0.0])
    retriever = EmbeddingRetriever(memory, _registry(provider), "ollama", "test-model")

    chunks = await retriever.retrieve("global", "q", top_k=5)
    assert chunks == []
