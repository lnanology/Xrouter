import pytest

from app.contracts.request import ChatCompletionRequest, ChatMessage
from app.execution.tool_loop import run_with_tools
from app.tools.registry import ToolRegistry
from tests.helpers import FakeTool, build_test_engine


def _request(content="what's the weather?"):
    return ChatCompletionRequest(model="auto", messages=[ChatMessage(role="user", content=content)])


def _tool_call(name: str, arguments: str = '{"query": "hi"}', call_id: str = "call_1") -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


# --- no executable tools -> falls through to a plain call --------------------

@pytest.mark.asyncio
async def test_no_enabled_names_falls_through_to_plain_handle_chat(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"content": "plain answer"}})
    response = await run_with_tools(engine, _request(), ToolRegistry(), enabled_names=[])
    assert response.choices[0].message["content"] == "plain answer"
    assert engine.ctx.providers.get("solo").call_count == 1


@pytest.mark.asyncio
async def test_requested_tool_not_registered_falls_through_to_plain_call(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"content": "plain answer"}})
    response = await run_with_tools(engine, _request(), ToolRegistry(), enabled_names=["web_search"])
    assert response.choices[0].message["content"] == "plain answer"


# --- single round-trip ---------------------------------------------------------

@pytest.mark.asyncio
async def test_single_tool_call_round_trip_feeds_result_back(tmp_path):
    tool = FakeTool(name="web_search", result="Paris is the capital of France.")
    registry = ToolRegistry()
    registry.register(tool)
    engine = await build_test_engine(tmp_path, {"solo": {"responses": [
        {"tool_calls": [_tool_call("web_search", '{"query": "capital of France"}')]},
        {"content": "The capital of France is Paris."},
    ]}})

    response = await run_with_tools(engine, _request("what's the capital of France?"), registry, enabled_names=["web_search"])

    assert response.choices[0].message["content"] == "The capital of France is Paris."
    assert tool.calls == [{"query": "capital of France"}]
    solo = engine.ctx.providers.get("solo")
    assert solo.call_count == 2
    # the tool's result must actually reach the model as a tool message
    tool_messages = [m for m in solo.last_request.messages if m.role == "tool"]
    assert tool_messages[0].content == "Paris is the capital of France."
    assert tool_messages[0].tool_call_id == "call_1"


# --- multiple iterations / max_iterations bound -------------------------------

@pytest.mark.asyncio
async def test_multiple_iterations_before_a_final_answer(tmp_path):
    tool = FakeTool(name="web_search", result="some search result")
    registry = ToolRegistry()
    registry.register(tool)
    engine = await build_test_engine(tmp_path, {"solo": {"responses": [
        {"tool_calls": [_tool_call("web_search", call_id="call_1")]},
        {"tool_calls": [_tool_call("web_search", call_id="call_2")]},
        {"content": "final answer"},
    ]}})

    response = await run_with_tools(engine, _request(), registry, enabled_names=["web_search"], max_iterations=5)

    assert response.choices[0].message["content"] == "final answer"
    assert len(tool.calls) == 2
    assert engine.ctx.providers.get("solo").call_count == 3


@pytest.mark.asyncio
async def test_max_iterations_bound_is_respected(tmp_path):
    tool = FakeTool(name="web_search", result="loop forever")
    registry = ToolRegistry()
    registry.register(tool)
    # The model always calls the tool again -- never emits a final answer.
    engine = await build_test_engine(tmp_path, {"solo": {
        "tool_calls": [_tool_call("web_search")],
    }})

    response = await run_with_tools(engine, _request(), registry, enabled_names=["web_search"], max_iterations=2)

    solo = engine.ctx.providers.get("solo")
    # initial call + exactly max_iterations follow-up calls, then it gives up
    assert solo.call_count == 3
    # still returns whatever the model last said rather than raising
    assert response.choices[0].message["tool_calls"]


# --- tool failures fed back rather than failing the node ---------------------

@pytest.mark.asyncio
async def test_tool_error_is_fed_back_as_the_tool_result_not_raised(tmp_path):
    tool = FakeTool(name="web_search", behavior="error")
    registry = ToolRegistry()
    registry.register(tool)
    engine = await build_test_engine(tmp_path, {"solo": {"responses": [
        {"tool_calls": [_tool_call("web_search")]},
        {"content": "I couldn't search, but here's my best guess."},
    ]}})

    response = await run_with_tools(engine, _request(), registry, enabled_names=["web_search"])

    assert response.choices[0].message["content"] == "I couldn't search, but here's my best guess."
    solo = engine.ctx.providers.get("solo")
    tool_messages = [m for m in solo.last_request.messages if m.role == "tool"]
    assert "Tool error" in tool_messages[0].content


# --- a tool call for something NOT enabled is left untouched ------------------

@pytest.mark.asyncio
async def test_tool_call_for_a_non_executable_name_is_returned_untouched(tmp_path):
    registry = ToolRegistry()
    registry.register(FakeTool(name="web_search"))
    engine = await build_test_engine(tmp_path, {"solo": {
        "tool_calls": [_tool_call("some_client_only_tool")],
    }})

    response = await run_with_tools(engine, _request(), registry, enabled_names=["web_search"])

    # not executed -- returned as-is for the client to handle itself
    assert response.choices[0].message["tool_calls"][0]["function"]["name"] == "some_client_only_tool"
    assert engine.ctx.providers.get("solo").call_count == 1
