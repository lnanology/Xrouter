import json as _json

import httpx
import pytest
import respx

from app.core.errors import ToolExecutionError, ToolUnavailableError
from app.tools.web_fetch import WebFetchTool


def _tool(api_key="test-key", base_url="https://api.tavily.com"):
    return WebFetchTool(api_key=api_key, base_url=base_url, timeout_seconds=5)


# --- configured / graceful degradation --------------------------------------

def test_configured_true_when_api_key_present():
    assert _tool(api_key="sk-1").configured is True


def test_configured_false_when_api_key_missing():
    assert _tool(api_key=None).configured is False


@pytest.mark.asyncio
async def test_execute_raises_tool_unavailable_when_not_configured():
    tool = _tool(api_key=None)
    with pytest.raises(ToolUnavailableError):
        await tool.execute({"url": "https://example.com"})
    await tool.close()


# --- argument validation ------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_raises_on_missing_url():
    tool = _tool()
    with pytest.raises(ToolExecutionError, match="url"):
        await tool.execute({})
    await tool.close()


@pytest.mark.asyncio
async def test_execute_raises_on_non_string_url():
    tool = _tool()
    with pytest.raises(ToolExecutionError, match="url"):
        await tool.execute({"url": 123})
    await tool.close()


# --- success / result shaping -------------------------------------------------

@pytest.mark.asyncio
async def test_execute_success_returns_page_content_with_url_prefix():
    tool = _tool()
    with respx.mock(base_url="https://api.tavily.com") as mock:
        mock.post("/extract").mock(return_value=httpx.Response(200, json={
            "results": [{"url": "https://example.com/page", "raw_content": "The page's full text."}],
        }))
        result = await tool.execute({"url": "https://example.com/page"})
    assert "https://example.com/page" in result
    assert "The page's full text." in result
    await tool.close()


@pytest.mark.asyncio
async def test_execute_sends_bearer_auth_header_and_url_in_body():
    tool = _tool(api_key="sk-secret")
    with respx.mock(base_url="https://api.tavily.com") as mock:
        route = mock.post("/extract").mock(return_value=httpx.Response(200, json={
            "results": [{"url": "https://example.com", "raw_content": "hi"}],
        }))
        await tool.execute({"url": "https://example.com"})
    sent = route.calls[0].request
    assert sent.headers["authorization"] == "Bearer sk-secret"
    body = _json.loads(sent.content)
    assert body["urls"] == "https://example.com"
    await tool.close()


@pytest.mark.asyncio
async def test_execute_truncates_very_long_content():
    tool = _tool()
    long_content = "x" * 20000
    with respx.mock(base_url="https://api.tavily.com") as mock:
        mock.post("/extract").mock(return_value=httpx.Response(200, json={
            "results": [{"url": "https://example.com", "raw_content": long_content}],
        }))
        result = await tool.execute({"url": "https://example.com"})
    assert len(result) < 20000
    await tool.close()


@pytest.mark.asyncio
async def test_execute_empty_raw_content_returns_friendly_text():
    tool = _tool()
    with respx.mock(base_url="https://api.tavily.com") as mock:
        mock.post("/extract").mock(return_value=httpx.Response(200, json={
            "results": [{"url": "https://example.com", "raw_content": ""}],
        }))
        result = await tool.execute({"url": "https://example.com"})
    assert "no extractable content" in result
    await tool.close()


@pytest.mark.asyncio
async def test_execute_reports_a_per_url_failure_from_failed_results():
    tool = _tool()
    with respx.mock(base_url="https://api.tavily.com") as mock:
        mock.post("/extract").mock(return_value=httpx.Response(200, json={
            "results": [],
            "failed_results": [{"url": "https://example.com/blocked", "error": "403 Forbidden"}],
        }))
        with pytest.raises(ToolExecutionError, match="403 Forbidden"):
            await tool.execute({"url": "https://example.com/blocked"})
    await tool.close()


@pytest.mark.asyncio
async def test_execute_raises_when_both_results_and_failed_results_are_empty():
    tool = _tool()
    with respx.mock(base_url="https://api.tavily.com") as mock:
        mock.post("/extract").mock(return_value=httpx.Response(200, json={"results": [], "failed_results": []}))
        with pytest.raises(ToolExecutionError, match="no result"):
            await tool.execute({"url": "https://example.com"})
    await tool.close()


# --- transport / HTTP failures ------------------------------------------------

@pytest.mark.asyncio
async def test_execute_non_200_raises_tool_execution_error():
    tool = _tool()
    with respx.mock(base_url="https://api.tavily.com") as mock:
        mock.post("/extract").mock(return_value=httpx.Response(500, text="internal error"))
        with pytest.raises(ToolExecutionError, match="500"):
            await tool.execute({"url": "https://example.com"})
    await tool.close()


@pytest.mark.asyncio
async def test_execute_timeout_raises_tool_execution_error():
    tool = _tool()
    with respx.mock(base_url="https://api.tavily.com") as mock:
        mock.post("/extract").mock(side_effect=httpx.TimeoutException("timed out"))
        with pytest.raises(ToolExecutionError, match="timed out"):
            await tool.execute({"url": "https://example.com"})
    await tool.close()


@pytest.mark.asyncio
async def test_execute_connection_error_raises_tool_execution_error():
    tool = _tool()
    with respx.mock(base_url="https://api.tavily.com") as mock:
        mock.post("/extract").mock(side_effect=httpx.ConnectError("connection refused"))
        with pytest.raises(ToolExecutionError, match="connection"):
            await tool.execute({"url": "https://example.com"})
    await tool.close()


def test_schema_declares_the_function_name_and_required_url():
    assert WebFetchTool(api_key="k").schema["function"]["name"] == "web_fetch"
    assert WebFetchTool(api_key="k").schema["function"]["parameters"]["required"] == ["url"]
