import pytest

from app.storage.database import Database
from app.storage.repositories.memory import MemoryRepository


@pytest.fixture
async def repo(tmp_path):
    db = Database(str(tmp_path / "test.sqlite3"))
    await db.init()
    return MemoryRepository(db)


@pytest.mark.asyncio
async def test_save_returns_the_stored_entry(repo):
    entry = await repo.save("global", "the sky is blue", tags=["fact"])
    assert entry.scope == "global"
    assert entry.content == "the sky is blue"
    assert entry.tags == ["fact"]
    assert entry.id.startswith("mem_")


@pytest.mark.asyncio
async def test_recent_returns_newest_first(repo):
    await repo.save("global", "first")
    await repo.save("global", "second")
    await repo.save("global", "third")
    recent = await repo.recent("global", limit=2)
    assert [e.content for e in recent] == ["third", "second"]


@pytest.mark.asyncio
async def test_recent_is_scoped(repo):
    await repo.save("scope-a", "a-entry")
    await repo.save("scope-b", "b-entry")
    assert [e.content for e in await repo.recent("scope-a", limit=10)] == ["a-entry"]
    assert [e.content for e in await repo.recent("scope-b", limit=10)] == ["b-entry"]


@pytest.mark.asyncio
async def test_search_matches_on_word_overlap(repo):
    await repo.save("global", "XRouter is an AI gateway written in Python")
    await repo.save("global", "the weather today is sunny and warm")
    results = await repo.search("global", "tell me about the gateway", limit=5)
    assert any("XRouter" in e.content for e in results)
    assert all("weather" not in e.content for e in results)


@pytest.mark.asyncio
async def test_search_scoped_to_the_given_scope(repo):
    await repo.save("scope-a", "the secret ingredient is basil")
    results = await repo.search("scope-b", "secret ingredient", limit=5)
    assert results == []


@pytest.mark.asyncio
async def test_search_with_no_meaningful_words_falls_back_to_recent(repo):
    await repo.save("global", "entry one")
    await repo.save("global", "entry two")
    # every word here is <=2 chars, so search() should fall back to recent()
    results = await repo.search("global", "a an", limit=5)
    assert [e.content for e in results] == ["entry two", "entry one"]


@pytest.mark.asyncio
async def test_search_returns_empty_list_when_nothing_matches(repo):
    await repo.save("global", "completely unrelated content")
    results = await repo.search("global", "quantum mechanics", limit=5)
    assert results == []
