import json

import pytest

from app.contracts.planner import PlanNodeSpec, PlanRequest, PlanSpec
from app.execution.dag import DagExecutor
from app.intelligence.planner import (
    PLAN_TOOL_NAME,
    PlannerError,
    _build_plan_tool_schema,
    _build_planning_request,
    _extract_tool_arguments,
    _parse_plan,
    _validate_plan_shape,
    generate_plan,
    to_dag_request,
)
from tests.helpers import build_test_engine


def _tool_call(arguments: str, name: str = PLAN_TOOL_NAME) -> dict:
    return {"id": "call_1", "type": "function", "function": {"name": name, "arguments": arguments}}


def _plan_json(nodes: list[dict]) -> str:
    return json.dumps({"nodes": nodes})


# --- pure request-building / parsing ----------------------------------------

def test_build_planning_request_forces_the_submit_plan_tool():
    req = _build_planning_request(PlanRequest(task="write a poem"), max_nodes=5)
    assert req.tool_choice == {"type": "function", "function": {"name": PLAN_TOOL_NAME}}
    assert req.tools[0]["function"]["name"] == PLAN_TOOL_NAME
    assert req.messages[-1].content == "write a poem"


def test_build_planning_request_includes_context_when_given():
    req = _build_planning_request(PlanRequest(task="do X", context="the user is a beginner"), max_nodes=5)
    joined = " ".join(m.content for m in req.messages if isinstance(m.content, str))
    assert "the user is a beginner" in joined


def test_extract_tool_arguments_returns_none_without_tool_calls():
    from app.contracts.response import ChatCompletionChoice, ChatCompletionResponse

    resp = ChatCompletionResponse(model="x", choices=[ChatCompletionChoice(message={"role": "assistant", "content": "hi"})])
    assert _extract_tool_arguments(resp) is None


def test_extract_tool_arguments_reads_the_first_tool_call():
    from app.contracts.response import ChatCompletionChoice, ChatCompletionResponse

    resp = ChatCompletionResponse(
        model="x",
        choices=[ChatCompletionChoice(message={"role": "assistant", "content": None, "tool_calls": [_tool_call('{"nodes": []}')]})],
    )
    assert _extract_tool_arguments(resp) == '{"nodes": []}'


def test_parse_plan_rejects_invalid_json():
    with pytest.raises(PlannerError, match="invalid JSON"):
        _parse_plan("not json at all")


def test_parse_plan_rejects_schema_violation():
    with pytest.raises(PlannerError, match="schema"):
        _parse_plan(json.dumps({"nodes": [{"id": "a"}]}))  # missing required "prompt"


def test_parse_plan_accepts_a_well_formed_plan():
    plan = _parse_plan(_plan_json([{"id": "a", "prompt": "hi"}]))
    assert plan.nodes == [PlanNodeSpec(id="a", prompt="hi")]


def test_parse_plan_coerces_null_depends_on_to_empty_list():
    plan = _parse_plan(json.dumps({"nodes": [{"id": "a", "prompt": "hi", "depends_on": None}]}))
    assert plan.nodes[0].depends_on == []


def test_validate_plan_shape_rejects_too_many_nodes():
    plan = PlanSpec(nodes=[PlanNodeSpec(id=f"n{i}", prompt="hi") for i in range(3)])
    with pytest.raises(PlannerError, match="exceeding"):
        _validate_plan_shape(plan, max_nodes=2)


def test_validate_plan_shape_rejects_a_cycle():
    plan = PlanSpec(nodes=[PlanNodeSpec(id="a", depends_on=["b"], prompt="hi"), PlanNodeSpec(id="b", depends_on=["a"], prompt="hi")])
    with pytest.raises(PlannerError, match="valid graph"):
        _validate_plan_shape(plan, max_nodes=20)


def test_validate_plan_shape_accepts_a_good_plan():
    plan = PlanSpec(nodes=[PlanNodeSpec(id="a", prompt="hi"), PlanNodeSpec(id="b", depends_on=["a"], prompt="hi")])
    _validate_plan_shape(plan, max_nodes=20)  # should not raise


def test_to_dag_request_carries_ids_deps_and_prompts_over():
    plan = PlanSpec(nodes=[PlanNodeSpec(id="a", prompt="do A"), PlanNodeSpec(id="b", depends_on=["a"], prompt="do B with {{a}}")])
    dag = to_dag_request(plan)
    assert [n.id for n in dag.nodes] == ["a", "b"]
    assert dag.nodes[1].depends_on == ["a"]
    assert dag.nodes[1].messages[0].content == "do B with {{a}}"


# --- end-to-end via a real ChatEngine (FakeProvider-backed) ------------------

@pytest.mark.asyncio
async def test_generate_plan_succeeds_on_the_first_try(tmp_path):
    plan_json = _plan_json([
        {"id": "capital", "prompt": "What is the capital of France?"},
        {"id": "translate", "depends_on": ["capital"], "prompt": "Translate {{capital}} to Spanish"},
    ])
    engine = await build_test_engine(tmp_path, {"solo": {"tool_calls": [_tool_call(plan_json)]}})
    plan_request = PlanRequest(task="tell me France's capital, translated", model="solo/test-model")

    plan, attempts = await generate_plan(engine, plan_request, max_nodes=20, max_retries=2)

    assert attempts == 1
    assert [n.id for n in plan.nodes] == ["capital", "translate"]


@pytest.mark.asyncio
async def test_generate_plan_retries_after_invalid_json_then_succeeds(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"responses": [
        {"tool_calls": [_tool_call("not valid json")]},
        {"tool_calls": [_tool_call(_plan_json([{"id": "a", "prompt": "hi"}]))]},
    ]}})
    plan_request = PlanRequest(task="do something", model="solo/test-model")

    plan, attempts = await generate_plan(engine, plan_request, max_nodes=20, max_retries=2)

    assert attempts == 2
    assert [n.id for n in plan.nodes] == ["a"]
    solo = engine.ctx.providers.get("solo")
    assert solo.call_count == 2
    # the retry must tell the model what specifically went wrong
    assert "invalid" in solo.last_request.messages[-1].content.lower()


@pytest.mark.asyncio
async def test_generate_plan_gives_up_after_exhausting_retries(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"tool_calls": [_tool_call("still not json")]}})
    plan_request = PlanRequest(task="do something", model="solo/test-model")

    with pytest.raises(PlannerError):
        await generate_plan(engine, plan_request, max_nodes=20, max_retries=1)

    solo = engine.ctx.providers.get("solo")
    assert solo.call_count == 2  # the original attempt plus exactly one retry


@pytest.mark.asyncio
async def test_generate_plan_rejects_a_response_with_no_tool_call(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"content": "I'll just answer directly: Paris."}})
    plan_request = PlanRequest(task="what's the capital of France?", model="solo/test-model")

    with pytest.raises(PlannerError, match="did not call submit_plan"):
        await generate_plan(engine, plan_request, max_nodes=20, max_retries=0)


@pytest.mark.asyncio
async def test_generate_plan_rejects_a_plan_exceeding_max_nodes(tmp_path):
    plan_json = _plan_json([{"id": f"n{i}", "prompt": "hi"} for i in range(3)])
    engine = await build_test_engine(tmp_path, {"solo": {"tool_calls": [_tool_call(plan_json)]}})
    plan_request = PlanRequest(task="do three things", model="solo/test-model")

    with pytest.raises(PlannerError, match="exceeding"):
        await generate_plan(engine, plan_request, max_nodes=2, max_retries=0)


# --- enable_tools / routing_policy schema + validation (Phase 3) ------------

def test_plan_tool_schema_offers_no_enable_tools_property_when_nothing_available():
    schema = _build_plan_tool_schema([])
    node_props = schema["function"]["parameters"]["properties"]["nodes"]["items"]["properties"]
    assert "enable_tools" not in node_props


def test_plan_tool_schema_enable_tools_enum_only_lists_available_tools():
    schema = _build_plan_tool_schema(["web_search"])
    node_props = schema["function"]["parameters"]["properties"]["nodes"]["items"]["properties"]
    assert node_props["enable_tools"]["items"]["enum"] == ["web_search"]


def test_plan_tool_schema_routing_policy_enum_is_always_offered():
    schema = _build_plan_tool_schema([])
    node_props = schema["function"]["parameters"]["properties"]["nodes"]["items"]["properties"]
    assert "balanced" in node_props["routing_policy"]["enum"]


def test_build_planning_request_mentions_available_tools_in_the_system_prompt():
    req = _build_planning_request(PlanRequest(task="research something"), max_nodes=5, available_tools=["web_search"])
    system = req.messages[0].content
    assert "web_search" in system


def test_build_planning_request_notes_no_tools_available():
    req = _build_planning_request(PlanRequest(task="do X"), max_nodes=5, available_tools=[])
    system = req.messages[0].content
    assert "No external tools" in system


def test_to_dag_request_carries_enable_tools_and_routing_policy_through():
    plan = PlanSpec(nodes=[PlanNodeSpec(id="a", prompt="search something", enable_tools=["web_search"], routing_policy="quality")])
    dag = to_dag_request(plan)
    assert dag.nodes[0].enable_tools == ["web_search"]
    assert dag.nodes[0].routing_policy == "quality"


def test_plan_tool_schema_always_offers_a_critique_property():
    schema = _build_plan_tool_schema([])
    node_props = schema["function"]["parameters"]["properties"]["nodes"]["items"]["properties"]
    assert node_props["critique"]["type"] == "boolean"


def test_to_dag_request_carries_critique_through():
    plan = PlanSpec(nodes=[PlanNodeSpec(id="a", prompt="synthesize the final answer", critique=True)])
    dag = to_dag_request(plan)
    assert dag.nodes[0].critique is True


def test_to_dag_request_critique_defaults_false():
    plan = PlanSpec(nodes=[PlanNodeSpec(id="a", prompt="hi")])
    dag = to_dag_request(plan)
    assert dag.nodes[0].critique is False


def test_validate_plan_shape_rejects_a_hallucinated_unavailable_tool():
    plan = PlanSpec(nodes=[PlanNodeSpec(id="a", prompt="hi", enable_tools=["not_a_real_tool"])])
    with pytest.raises(PlannerError, match="unknown/unavailable tool"):
        _validate_plan_shape(plan, max_nodes=20, available_tools=["web_search"])


def test_validate_plan_shape_accepts_a_tool_that_is_available():
    plan = PlanSpec(nodes=[PlanNodeSpec(id="a", prompt="hi", enable_tools=["web_search"])])
    _validate_plan_shape(plan, max_nodes=20, available_tools=["web_search"])  # should not raise


def test_validate_plan_shape_rejects_any_tool_when_none_are_available():
    plan = PlanSpec(nodes=[PlanNodeSpec(id="a", prompt="hi", enable_tools=["web_search"])])
    with pytest.raises(PlannerError, match="unknown/unavailable tool"):
        _validate_plan_shape(plan, max_nodes=20, available_tools=[])


@pytest.mark.asyncio
async def test_generate_plan_offers_only_registered_configured_tools(tmp_path):
    plan_json = _plan_json([{"id": "a", "prompt": "search for X", "enable_tools": ["web_search"]}])
    engine = await build_test_engine(
        tmp_path, {"solo": {"tool_calls": [_tool_call(plan_json)]}},
        tool_specs={"web_search": {}},
    )
    plan_request = PlanRequest(task="search for X", model="solo/test-model")

    plan, attempts = await generate_plan(engine, plan_request, max_nodes=20, max_retries=0)

    assert attempts == 1
    assert plan.nodes[0].enable_tools == ["web_search"]
    solo = engine.ctx.providers.get("solo")
    sent_schema = solo.last_request.tools[0]
    node_props = sent_schema["function"]["parameters"]["properties"]["nodes"]["items"]["properties"]
    assert node_props["enable_tools"]["items"]["enum"] == ["web_search"]


@pytest.mark.asyncio
async def test_generate_plan_rejects_a_hallucinated_tool_and_retries(tmp_path):
    bad_plan = _plan_json([{"id": "a", "prompt": "hi", "enable_tools": ["fake_tool"]}])
    good_plan = _plan_json([{"id": "a", "prompt": "hi"}])
    engine = await build_test_engine(tmp_path, {"solo": {"responses": [
        {"tool_calls": [_tool_call(bad_plan)]},
        {"tool_calls": [_tool_call(good_plan)]},
    ]}})
    plan_request = PlanRequest(task="do something", model="solo/test-model")

    plan, attempts = await generate_plan(engine, plan_request, max_nodes=20, max_retries=1)

    assert attempts == 2
    assert plan.nodes[0].enable_tools == []
    solo = engine.ctx.providers.get("solo")
    assert "unavailable tool" in solo.last_request.messages[-1].content.lower()


@pytest.mark.asyncio
async def test_plan_then_execute_runs_the_generated_dag_end_to_end(tmp_path):
    plan_json = _plan_json([
        {"id": "capital", "prompt": "What is the capital of France?"},
        {"id": "translate", "depends_on": ["capital"], "prompt": "Translate {{capital}} to Spanish"},
    ])
    engine = await build_test_engine(tmp_path, {"solo": {"responses": [
        {"tool_calls": [_tool_call(plan_json)]},
        {"content": "Paris"},
        {"content": "Madrid (as a translation exercise)"},
    ]}})
    plan_request = PlanRequest(task="tell me France's capital, translated", model="solo/test-model")

    plan, attempts = await generate_plan(engine, plan_request, max_nodes=20, max_retries=2)
    dag_request = to_dag_request(plan)
    # Every DAG node inherits model="auto" from DagNodeRequest's default,
    # and the only registered provider is "solo" -- so the generated plan
    # actually gets executed through the same engine, not just parsed.
    result = await DagExecutor(engine).run(dag_request)

    assert attempts == 1
    assert result.status == "success"
    assert [n.status for n in result.nodes] == ["success", "success"]
    solo = engine.ctx.providers.get("solo")
    # The DAG executor's own {{node_id}} substitution must have received
    # the *planner's own* dependency wiring, not something re-derived.
    assert solo.last_request.messages[0].content == "Translate Paris to Spanish"
