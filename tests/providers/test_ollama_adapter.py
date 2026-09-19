import httpx
import pytest
import respx

from app.contracts.provider import ProviderConfig, ProviderStatus
from app.contracts.request import ChatCompletionRequest, ChatMessage
from app.core.errors import ProviderConnectionError
from app.providers.ollama.adapter import OllamaAdapter


def _config():
    return ProviderConfig(id="ollama", name="ollama", type="ollama", base_url="http://localhost:11434", timeout_seconds=5)


@pytest.mark.asyncio
async def test_health_offline_when_unreachable():
    adapter = OllamaAdapter(_config())
    with respx.mock(base_url="http://localhost:11434") as mock:
        mock.get("/api/tags").mock(side_effect=httpx.ConnectError("refused"))
        health = await adapter.health()
    assert health.status == ProviderStatus.OFFLINE
    await adapter.close()


@pytest.mark.asyncio
async def test_health_healthy_when_reachable():
    adapter = OllamaAdapter(_config())
    with respx.mock(base_url="http://localhost:11434") as mock:
        mock.get("/api/tags").mock(return_value=httpx.Response(200, json={"models": []}))
        health = await adapter.health()
    assert health.status == ProviderStatus.HEALTHY
    await adapter.close()


@pytest.mark.asyncio
async def test_list_models_returns_empty_on_failure_not_raise():
    adapter = OllamaAdapter(_config())
    with respx.mock(base_url="http://localhost:11434") as mock:
        mock.get("/api/tags").mock(side_effect=httpx.ConnectError("refused"))
        models = await adapter.list_models()
    assert models == []
    await adapter.close()


@pytest.mark.asyncio
async def test_list_models_parses_response():
    adapter = OllamaAdapter(_config())
    with respx.mock(base_url="http://localhost:11434") as mock:
        mock.get("/api/tags").mock(return_value=httpx.Response(200, json={"models": [{"name": "llama3"}]}))
        models = await adapter.list_models()
    assert len(models) == 1
    assert models[0].name == "llama3"
    assert models[0].cost_score == 1.0  # local = free
    await adapter.close()


@pytest.mark.asyncio
async def test_chat_success():
    adapter = OllamaAdapter(_config())
    with respx.mock(base_url="http://localhost:11434") as mock:
        mock.post("/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={
                    "message": {"role": "assistant", "content": "hi there"},
                    "prompt_eval_count": 3,
                    "eval_count": 2,
                },
            )
        )
        req = ChatCompletionRequest(model="llama3", messages=[ChatMessage(role="user", content="hi")])
        resp = await adapter.chat("llama3", req)
    assert resp.choices[0].message["content"] == "hi there"
    assert resp.usage.total_tokens == 5
    await adapter.close()


@pytest.mark.asyncio
async def test_chat_connection_error_raises_typed_error():
    adapter = OllamaAdapter(_config())
    with respx.mock(base_url="http://localhost:11434") as mock:
        mock.post("/api/chat").mock(side_effect=httpx.ConnectError("refused"))
        req = ChatCompletionRequest(model="llama3", messages=[ChatMessage(role="user", content="hi")])
        with pytest.raises(ProviderConnectionError):
            await adapter.chat("llama3", req)
    await adapter.close()


@pytest.mark.asyncio
async def test_capabilities_includes_embeddings():
    adapter = OllamaAdapter(_config())
    from app.contracts.provider import ProviderCapability

    assert ProviderCapability.EMBEDDINGS in adapter.capabilities()
    await adapter.close()


@pytest.mark.asyncio
async def test_embed_success():
    adapter = OllamaAdapter(_config())
    with respx.mock(base_url="http://localhost:11434") as mock:
        mock.post("/api/embed").mock(return_value=httpx.Response(200, json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]}))
        vectors = await adapter.embed("nomic-embed-text", ["a", "b"])
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    await adapter.close()


@pytest.mark.asyncio
async def test_embed_connection_error_raises_typed_error():
    adapter = OllamaAdapter(_config())
    with respx.mock(base_url="http://localhost:11434") as mock:
        mock.post("/api/embed").mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(ProviderConnectionError):
            await adapter.embed("nomic-embed-text", ["a"])
    await adapter.close()
