import aiosqlite
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


# --- embedding_json round-trip / embedded() ----------------------------------

@pytest.mark.asyncio
async def test_save_and_recent_round_trip_an_embedding(repo):
    await repo.save("global", "has an embedding", embedding=[0.1, 0.2, 0.3])
    recent = await repo.recent("global", limit=1)
    assert recent[0].embedding == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_save_without_embedding_leaves_it_none(repo):
    await repo.save("global", "no embedding")
    recent = await repo.recent("global", limit=1)
    assert recent[0].embedding is None


@pytest.mark.asyncio
async def test_embedded_only_returns_entries_with_an_embedding(repo):
    await repo.save("global", "embedded one", embedding=[1.0, 0.0])
    await repo.save("global", "not embedded")
    entries = await repo.embedded("global")
    assert [e.content for e in entries] == ["embedded one"]


@pytest.mark.asyncio
async def test_embedded_is_scoped_and_newest_first(repo):
    await repo.save("scope-a", "a-old", embedding=[1.0])
    await repo.save("scope-a", "a-new", embedding=[1.0])
    await repo.save("scope-b", "b-entry", embedding=[1.0])
    entries = await repo.embedded("scope-a")
    assert [e.content for e in entries] == ["a-new", "a-old"]


@pytest.mark.asyncio
async def test_embedded_respects_limit(repo):
    for i in range(5):
        await repo.save("global", f"entry {i}", embedding=[1.0])
    entries = await repo.embedded("global", limit=2)
    assert len(entries) == 2


@pytest.mark.asyncio
async def test_database_init_adds_embedding_json_to_a_pre_existing_table(tmp_path):
    """Simulates an upgrade: a real data/xrouter.sqlite3 that was already
    initialized before embedding_json existed must not break the moment
    save() tries to insert into the new column."""
    db_path = str(tmp_path / "old.sqlite3")
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(
            "CREATE TABLE memory_entries (id TEXT PRIMARY KEY, scope TEXT NOT NULL, content TEXT NOT NULL,"
            " tags_json TEXT NOT NULL, created_at REAL NOT NULL)"
        )
        await conn.commit()

    db = Database(db_path)
    await db.init()  # must not raise, and must add the missing column
    repo = MemoryRepository(db)
    entry = await repo.save("global", "after upgrade", embedding=[1.0, 2.0])
    assert entry.embedding == [1.0, 2.0]
    recent = await repo.recent("global", limit=1)
    assert recent[0].content == "after upgrade"
    assert recent[0].embedding == [1.0, 2.0]
