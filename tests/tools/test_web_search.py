import httpx
import pytest
import respx

from app.core.errors import ToolExecutionError, ToolUnavailableError
from app.tools.web_search import WebSearchTool


def _tool(api_key="test-key", base_url="https://api.tavily.com"):
    return WebSearchTool(api_key=api_key, base_url=base_url, timeout_seconds=5)


# --- configured / graceful degradation --------------------------------------

def test_configured_true_when_api_key_present():
    assert _tool(api_key="sk-1").configured is True


def test_configured_false_when_api_key_missing():
    assert _tool(api_key=None).configured is False


@pytest.mark.asyncio
async def test_execute_raises_tool_unavailable_when_not_configured():
    tool = _tool(api_key=None)
    with pytest.raises(ToolUnavailableError):
        await tool.execute({"query": "xrouter"})
    await tool.close()


# --- argument validation ------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_raises_on_missing_query():
    tool = _tool()
    with pytest.raises(ToolExecutionError, match="query"):
        await tool.execute({})
    await tool.close()


@pytest.mark.asyncio
async def test_execute_raises_on_non_string_query():
    tool = _tool()
    with pytest.raises(ToolExecutionError, match="query"):
        await tool.execute({"query": 123})
    await tool.close()


# --- success / result shaping -------------------------------------------------

@pytest.mark.asyncio
async def test_execute_success_formats_results_as_text():
    tool = _tool()
    with respx.mock(base_url="https://api.tavily.com") as mock:
        mock.post("/search").mock(return_value=httpx.Response(200, json={
            "results": [
                {"title": "XRouter", "url": "https://example.com/xrouter", "content": "An AI gateway."},
                {"title": "Second", "url": "https://example.com/second", "content": "Another result."},
            ]
        }))
        result = await tool.execute({"query": "xrouter"})
    assert "XRouter" in result
    assert "https://example.com/xrouter" in result
    assert "Second" in result
    await tool.close()


@pytest.mark.asyncio
async def test_execute_no_results_returns_friendly_text():
    tool = _tool()
    with respx.mock(base_url="https://api.tavily.com") as mock:
        mock.post("/search").mock(return_value=httpx.Response(200, json={"results": []}))
        result = await tool.execute({"query": "something obscure"})
    assert "No results" in result
    await tool.close()


@pytest.mark.asyncio
async def test_execute_clamps_max_results_within_bounds():
    tool = _tool()
    with respx.mock(base_url="https://api.tavily.com") as mock:
        route = mock.post("/search").mock(return_value=httpx.Response(200, json={"results": []}))
        await tool.execute({"query": "hi", "max_results": 999})
    sent = route.calls[0].request
    import json as _json

    body = _json.loads(sent.content)
    assert body["max_results"] == 10  # clamped to the documented max
    await tool.close()


@pytest.mark.asyncio
async def test_execute_invalid_max_results_falls_back_to_default():
    tool = _tool()
    with respx.mock(base_url="https://api.tavily.com") as mock:
        route = mock.post("/search").mock(return_value=httpx.Response(200, json={"results": []}))
        await tool.execute({"query": "hi", "max_results": "not a number"})
    import json as _json

    body = _json.loads(route.calls[0].request.content)
    assert body["max_results"] == 5
    await tool.close()


# --- transport / HTTP failures ------------------------------------------------

@pytest.mark.asyncio
async def test_execute_non_200_raises_tool_execution_error():
    tool = _tool()
    with respx.mock(base_url="https://api.tavily.com") as mock:
        mock.post("/search").mock(return_value=httpx.Response(500, text="internal error"))
        with pytest.raises(ToolExecutionError, match="500"):
            await tool.execute({"query": "hi"})
    await tool.close()


@pytest.mark.asyncio
async def test_execute_timeout_raises_tool_execution_error():
    tool = _tool()
    with respx.mock(base_url="https://api.tavily.com") as mock:
        mock.post("/search").mock(side_effect=httpx.TimeoutException("timed out"))
        with pytest.raises(ToolExecutionError, match="timed out"):
            await tool.execute({"query": "hi"})
    await tool.close()


@pytest.mark.asyncio
async def test_execute_connection_error_raises_tool_execution_error():
    tool = _tool()
    with respx.mock(base_url="https://api.tavily.com") as mock:
        mock.post("/search").mock(side_effect=httpx.ConnectError("connection refused"))
        with pytest.raises(ToolExecutionError, match="connection"):
            await tool.execute({"query": "hi"})
    await tool.close()


def test_schema_declares_the_function_name_and_required_query():
    assert WebSearchTool(api_key="k").schema["function"]["name"] == "web_search"
    assert WebSearchTool(api_key="k").schema["function"]["parameters"]["required"] == ["query"]
