import pytest

from app.agents.researcher import _build_research_request, research
from app.tools.registry import ToolRegistry
from tests.helpers import build_test_engine


# --- pure request-building ---------------------------------------------------

def test_build_research_request_mentions_the_tool_when_available():
    req = _build_research_request("what is the latest on X?", has_search=True, routing_policy=None)
    system = req.messages[0].content
    assert "web_search tool" in system
    assert req.messages[-1].content == "what is the latest on X?"


def test_build_research_request_admits_no_tool_when_unavailable():
    req = _build_research_request("what is the latest on X?", has_search=False, routing_policy=None)
    system = req.messages[0].content
    assert "No web search tool is available" in system


# --- end-to-end via a real ChatEngine (FakeProvider-backed) ------------------

@pytest.mark.asyncio
async def test_research_answers_from_own_knowledge_when_no_search_tool(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"content": "Paris is the capital of France."}})
    result = await research(engine, "what is the capital of France?", ToolRegistry())
    assert result.tool_available is False
    assert result.answer == "Paris is the capital of France."


@pytest.mark.asyncio
async def test_research_uses_web_search_when_available(tmp_path):
    engine = await build_test_engine(
        tmp_path,
        {"solo": {"responses": [
            {"tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "web_search", "arguments": '{"query": "latest X news"}'},
            }]},
            {"content": "Based on the search, here's the latest on X."},
        ]}},
        tool_specs={"web_search": {"result": "some search result"}},
    )
    result = await research(engine, "what's the latest on X?", engine.ctx.tools)
    assert result.tool_available is True
    assert result.answer == "Based on the search, here's the latest on X."
    tool = engine.ctx.tools.get("web_search")
    assert tool.calls == [{"query": "latest X news"}]


@pytest.mark.asyncio
async def test_research_tool_available_false_when_tool_registered_but_unconfigured(tmp_path):
    engine = await build_test_engine(
        tmp_path, {"solo": {"content": "answering without search"}},
        tool_specs={"web_search": {"configured": False}},
    )
    result = await research(engine, "anything current?", engine.ctx.tools)
    assert result.tool_available is False
    assert result.answer == "answering without search"
