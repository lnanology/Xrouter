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


# --- Tool calling (real, verified against ollama/ollama's own docs/api.md) ---


@pytest.mark.asyncio
async def test_chat_forwards_tools_in_request_body():
    adapter = OllamaAdapter(_config())
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}}}]
    with respx.mock(base_url="http://localhost:11434") as mock:
        route = mock.post("/api/chat").mock(
            return_value=httpx.Response(200, json={"message": {"role": "assistant", "content": ""}})
        )
        req = ChatCompletionRequest(model="llama3.2", messages=[ChatMessage(role="user", content="weather?")], tools=tools)
        await adapter.chat("llama3.2", req)
    assert route.calls.last.request.content
    import json as _json

    sent = _json.loads(route.calls.last.request.content)
    assert sent["tools"] == tools
    await adapter.close()


@pytest.mark.asyncio
async def test_chat_omits_tools_key_when_no_tools_requested():
    adapter = OllamaAdapter(_config())
    with respx.mock(base_url="http://localhost:11434") as mock:
        route = mock.post("/api/chat").mock(return_value=httpx.Response(200, json={"message": {"role": "assistant", "content": "hi"}}))
        req = ChatCompletionRequest(model="llama3.2", messages=[ChatMessage(role="user", content="hi")])
        await adapter.chat("llama3.2", req)
    import json as _json

    sent = _json.loads(route.calls.last.request.content)
    assert "tools" not in sent
    await adapter.close()


@pytest.mark.asyncio
async def test_chat_response_tool_call_arguments_become_a_json_string():
    """Ollama's own wire format returns tool_calls[].function.arguments as a
    native dict; XRouter must hand callers a genuinely OpenAI-compatible
    response, where arguments is a JSON-encoded string."""
    adapter = OllamaAdapter(_config())
    with respx.mock(base_url="http://localhost:11434") as mock:
        mock.post("/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Tokyo"}}}],
                    }
                },
            )
        )
        req = ChatCompletionRequest(model="llama3.2", messages=[ChatMessage(role="user", content="weather in tokyo?")])
        resp = await adapter.chat("llama3.2", req)
    tool_calls = resp.choices[0].message["tool_calls"]
    assert isinstance(tool_calls[0]["function"]["arguments"], str)
    import json as _json

    assert _json.loads(tool_calls[0]["function"]["arguments"]) == {"city": "Tokyo"}
    await adapter.close()


@pytest.mark.asyncio
async def test_chat_replays_prior_assistant_tool_call_history_in_ollamas_dict_format():
    """A client replaying a prior turn sends OpenAI-style tool_calls (a JSON
    string for arguments); Ollama's /api/chat expects a native dict there."""
    adapter = OllamaAdapter(_config())
    import json as _json

    history = [
        ChatMessage(role="user", content="weather in Toronto?"),
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=[{"function": {"name": "get_weather", "arguments": _json.dumps({"city": "Toronto"})}}],
        ),
        ChatMessage(role="tool", content="11 degrees celsius", name="get_weather", tool_call_id="call_1"),
    ]
    with respx.mock(base_url="http://localhost:11434") as mock:
        route = mock.post("/api/chat").mock(return_value=httpx.Response(200, json={"message": {"role": "assistant", "content": "It's 11C."}}))
        req = ChatCompletionRequest(model="llama3.2", messages=history)
        await adapter.chat("llama3.2", req)
    sent = _json.loads(route.calls.last.request.content)
    assert sent["messages"][1]["tool_calls"][0]["function"]["arguments"] == {"city": "Toronto"}
    assert sent["messages"][2]["tool_name"] == "get_weather"
    await adapter.close()


@pytest.mark.asyncio
async def test_list_models_sets_supports_tools_true_when_capability_present():
    adapter = OllamaAdapter(_config())
    with respx.mock(base_url="http://localhost:11434") as mock:
        mock.get("/api/tags").mock(return_value=httpx.Response(200, json={"models": [{"name": "llama3.2:3b"}]}))
        mock.post("/api/show").mock(return_value=httpx.Response(200, json={"capabilities": ["completion", "tools"]}))
        models = await adapter.list_models()
    assert models[0].supports_tools is True
    await adapter.close()


@pytest.mark.asyncio
async def test_list_models_sets_supports_tools_false_when_capability_absent():
    adapter = OllamaAdapter(_config())
    with respx.mock(base_url="http://localhost:11434") as mock:
        mock.get("/api/tags").mock(return_value=httpx.Response(200, json={"models": [{"name": "llava"}]}))
        mock.post("/api/show").mock(return_value=httpx.Response(200, json={"capabilities": ["completion", "vision"]}))
        models = await adapter.list_models()
    assert models[0].supports_tools is False
    await adapter.close()


@pytest.mark.asyncio
async def test_list_models_fails_open_to_no_tool_support_if_show_errors():
    adapter = OllamaAdapter(_config())
    with respx.mock(base_url="http://localhost:11434") as mock:
        mock.get("/api/tags").mock(return_value=httpx.Response(200, json={"models": [{"name": "llama3.2:3b"}]}))
        mock.post("/api/show").mock(side_effect=httpx.ConnectError("refused"))
        models = await adapter.list_models()
    assert models[0].supports_tools is False
    await adapter.close()
