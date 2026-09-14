import tempfile
from pathlib import Path

import pytest

from app.cache.manager import CacheManager
from app.contracts.request import ChatCompletionRequest, ChatMessage
from app.core.config import CacheConfig


@pytest.fixture
def cache_manager():
    with tempfile.TemporaryDirectory() as d:
        db_path = str(Path(d) / "test.sqlite3")
        import asyncio

        from app.storage.database import Database

        db = Database(db_path)
        asyncio.get_event_loop().run_until_complete(db.init())
        yield CacheManager(CacheConfig(enabled=True, l1_max_entries=10, default_ttl_seconds=60), db_path)


def _req(text="what is 2+2", stream=False):
    return ChatCompletionRequest(model="auto", messages=[ChatMessage(role="user", content=text)], stream=stream)


def test_streaming_requests_not_cacheable(cache_manager):
    assert cache_manager.is_cacheable(_req(stream=True)) is False


def test_volatile_keyword_not_cacheable(cache_manager):
    assert cache_manager.is_cacheable(_req("what is the latest stock price of AAPL")) is False


def test_plain_request_is_cacheable(cache_manager):
    assert cache_manager.is_cacheable(_req("what is 2+2")) is True


@pytest.mark.asyncio
async def test_cache_set_and_get_roundtrip(cache_manager):
    req = _req("hello")
    key = cache_manager.key_for(req)
    assert await cache_manager.get(key) is None
    await cache_manager.set(key, {"answer": 42})
    result = await cache_manager.get(key)
    assert result == {"answer": 42}


def test_same_request_same_key(cache_manager):
    a = cache_manager.key_for(_req("hello"))
    b = cache_manager.key_for(_req("hello"))
    c = cache_manager.key_for(_req("different"))
    assert a == b
    assert a != c
