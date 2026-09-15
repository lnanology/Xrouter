import pytest

from app.contracts.dag import DagNodeRequest, DagRunRequest
from app.execution.dag import DagExecutor, DagValidationError, _substitute, _validate_and_order
from app.tools.registry import ToolRegistry
from tests.helpers import FakeTool, build_test_engine


def _node(id_, depends_on=None, content="hi", model="auto", enable_tools=None, critique=False):
    return DagNodeRequest(
        id=id_, depends_on=depends_on or [], messages=[{"role": "user", "content": content}],
        model=model, enable_tools=enable_tools or [], critique=critique,
    )


# --- pure validation/ordering -----------------------------------------------

def test_linear_chain_orders_into_separate_waves():
    waves = _validate_and_order([_node("a"), _node("b", ["a"]), _node("c", ["b"])], max_nodes=20)
    assert [[n.id for n in w] for w in waves] == [["a"], ["b"], ["c"]]


def test_independent_nodes_share_a_wave():
    waves = _validate_and_order([_node("a"), _node("b"), _node("c", ["a", "b"])], max_nodes=20)
    assert {n.id for n in waves[0]} == {"a", "b"}
    assert [n.id for n in waves[1]] == ["c"]


def test_empty_dag_rejected():
    with pytest.raises(DagValidationError):
        _validate_and_order([], max_nodes=20)


def test_too_many_nodes_rejected():
    nodes = [_node(f"n{i}") for i in range(5)]
    with pytest.raises(DagValidationError, match="exceeding"):
        _validate_and_order(nodes, max_nodes=3)


def test_duplicate_id_rejected():
    with pytest.raises(DagValidationError, match="Duplicate"):
        _validate_and_order([_node("a"), _node("a")], max_nodes=20)


def test_self_dependency_rejected():
    with pytest.raises(DagValidationError, match="depend on itself"):
        _validate_and_order([_node("a", ["a"])], max_nodes=20)


def test_unknown_dependency_rejected():
    with pytest.raises(DagValidationError, match="unknown node"):
        _validate_and_order([_node("a", ["ghost"])], max_nodes=20)


def test_cycle_rejected():
    with pytest.raises(DagValidationError, match="cycle"):
        _validate_and_order([_node("a", ["b"]), _node("b", ["a"])], max_nodes=20)


# --- pure substitution -------------------------------------------------------

def test_substitute_replaces_known_placeholder():
    assert _substitute("hello {{a}} world", {"a": "X"}) == "hello X world"


def test_substitute_leaves_unknown_placeholder_untouched():
    assert _substitute("hello {{ghost}}", {}) == "hello {{ghost}}"


def test_substitute_handles_multiple_and_repeated_placeholders():
    assert _substitute("{{a}}-{{b}}-{{a}}", {"a": "1", "b": "2"}) == "1-2-1"


# --- end-to-end via a real ChatEngine (FakeProvider-backed) -----------------

@pytest.mark.asyncio
async def test_linear_chain_substitutes_upstream_output(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"content": "Paris"}})
    dag = DagRunRequest(nodes=[
        _node("capital", content="What is the capital of France?"),
        _node("translate", depends_on=["capital"], content="Translate {{capital}} to Spanish"),
    ])
    result = await DagExecutor(engine).run(dag)

    assert result.status == "success"
    assert [n.status for n in result.nodes] == ["success", "success"]
    solo = engine.ctx.providers.get("solo")
    assert solo.last_request.messages[0].content == "Translate Paris to Spanish"


@pytest.mark.asyncio
async def test_parallel_fan_in_combines_both_upstream_outputs(tmp_path):
    engine = await build_test_engine(tmp_path, {
        "num5": {"content": "5"}, "num3": {"content": "3"}, "combiner": {"content": "8"},
    })
    dag = DagRunRequest(nodes=[
        _node("a", content="pick a number", model="num5/test-model"),
        _node("b", content="pick a number", model="num3/test-model"),
        _node("c", depends_on=["a", "b"], content="combine {{a}} and {{b}}", model="combiner/test-model"),
    ])
    result = await DagExecutor(engine).run(dag)

    assert result.status == "success"
    combiner = engine.ctx.providers.get("combiner")
    assert combiner.last_request.messages[0].content == "combine 5 and 3"


@pytest.mark.asyncio
async def test_failed_node_cascades_skip_to_dependents_but_not_siblings(tmp_path):
    engine = await build_test_engine(tmp_path, {
        "good": {"content": "ok"}, "bad": {"behavior": "server_error"},
    })
    dag = DagRunRequest(nodes=[
        _node("a", content="hi", model="bad/test-model"),
        _node("b", depends_on=["a"], content="use {{a}}", model="good/test-model"),
        _node("d", content="independent", model="good/test-model"),
    ])
    result = await DagExecutor(engine).run(dag)

    statuses = {n.id: n.status for n in result.nodes}
    assert statuses == {"a": "failed", "b": "skipped", "d": "success"}
    assert result.status == "partial"


@pytest.mark.asyncio
async def test_all_nodes_failing_reports_overall_failed(tmp_path):
    engine = await build_test_engine(tmp_path, {"bad": {"behavior": "server_error"}})
    dag = DagRunRequest(nodes=[_node("a", model="bad/test-model")])
    result = await DagExecutor(engine).run(dag)

    assert result.status == "failed"
    assert result.nodes[0].status == "failed"


@pytest.mark.asyncio
async def test_max_nodes_enforced_end_to_end(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"content": "ok"}})
    dag = DagRunRequest(nodes=[_node("a"), _node("b")])
    with pytest.raises(DagValidationError):
        await DagExecutor(engine, max_nodes=1).run(dag)


# --- enable_tools (Phase 3: XRouter-executed tool loop) ----------------------

def _tool_call(name: str, arguments: str = '{"query": "hi"}') -> dict:
    return {"id": "call_1", "type": "function", "function": {"name": name, "arguments": arguments}}


@pytest.mark.asyncio
async def test_node_with_enable_tools_invokes_the_tool_and_reaches_final_response(tmp_path):
    engine = await build_test_engine(
        tmp_path, {"solo": {"responses": [
            {"tool_calls": [_tool_call("web_search", '{"query": "capital of France"}')]},
            {"content": "Paris, per the search."},
        ]}},
        tool_specs={"web_search": {"result": "Paris is the capital of France."}},
    )
    dag = DagRunRequest(nodes=[_node("a", content="what's the capital of France?", enable_tools=["web_search"])])
    result = await DagExecutor(
        engine, tools=engine.ctx.tools, max_tool_iterations=3,
    ).run(dag)

    assert result.status == "success"
    assert "Paris" in result.nodes[0].response.choices[0].message["content"]
    tool = engine.ctx.tools.get("web_search")
    assert tool.calls == [{"query": "capital of France"}]


@pytest.mark.asyncio
async def test_node_without_enable_tools_never_touches_the_registry(tmp_path):
    engine = await build_test_engine(
        tmp_path, {"solo": {"content": "plain answer, no tools"}},
        tool_specs={"web_search": {"result": "should not be called"}},
    )
    dag = DagRunRequest(nodes=[_node("a", content="hi")])
    result = await DagExecutor(engine, tools=engine.ctx.tools).run(dag)

    assert result.status == "success"
    tool = engine.ctx.tools.get("web_search")
    assert tool.calls == []


@pytest.mark.asyncio
async def test_node_requesting_an_unregistered_tool_falls_back_to_a_plain_call(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"content": "plain answer"}})
    dag = DagRunRequest(nodes=[_node("a", content="hi", enable_tools=["web_search"])])
    # no tools= passed at all -- DagExecutor defaults to an empty registry
    result = await DagExecutor(engine).run(dag)

    assert result.status == "success"
    assert result.nodes[0].response.choices[0].message["content"] == "plain answer"


# --- critique (Phase 3: per-node review) --------------------------------------

def _critique_tool_call(satisfied: bool, feedback: str = "") -> dict:
    import json as _json

    return {
        "id": "call_1", "type": "function",
        "function": {"name": "submit_critique", "arguments": _json.dumps({"satisfied": satisfied, "feedback": feedback})},
    }


@pytest.mark.asyncio
async def test_node_without_critique_never_triggers_a_review_call(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"content": "plain answer"}})
    result = await DagExecutor(engine).run(DagRunRequest(nodes=[_node("a", content="hi")]))

    assert result.status == "success"
    assert result.nodes[0].critique is None
    assert engine.ctx.providers.get("solo").call_count == 1  # no review call at all


@pytest.mark.asyncio
async def test_node_with_critique_satisfied_first_try(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"responses": [
        {"content": "a good answer"},
        {"tool_calls": [_critique_tool_call(True)]},
    ]}})
    result = await DagExecutor(engine).run(DagRunRequest(nodes=[_node("a", content="hi", critique=True)]))

    assert result.status == "success"
    assert result.nodes[0].critique.satisfied is True
    assert result.nodes[0].response.choices[0].message["content"] == "a good answer"


@pytest.mark.asyncio
async def test_node_with_critique_unsatisfied_then_satisfied_reruns_the_node(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"responses": [
        {"content": "a weak answer"},
        {"tool_calls": [_critique_tool_call(False, "needs more detail")]},
        {"content": "a much better answer"},
        {"tool_calls": [_critique_tool_call(True)]},
    ]}})
    result = await DagExecutor(engine, max_critique_retries=1).run(
        DagRunRequest(nodes=[_node("a", content="hi", critique=True)])
    )

    assert result.status == "success"
    assert result.nodes[0].critique.satisfied is True
    assert result.nodes[0].response.choices[0].message["content"] == "a much better answer"
    solo = engine.ctx.providers.get("solo")
    assert solo.call_count == 4  # node + critique + rerun + critique


@pytest.mark.asyncio
async def test_node_with_critique_gives_up_after_exhausting_retries_but_still_succeeds(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"responses": [
        {"content": "attempt 1"},
        {"tool_calls": [_critique_tool_call(False, "nope")]},
        {"content": "attempt 2"},
        {"tool_calls": [_critique_tool_call(False, "still nope")]},
    ]}})
    result = await DagExecutor(engine, max_critique_retries=1).run(
        DagRunRequest(nodes=[_node("a", content="hi", critique=True)])
    )

    # The node itself still succeeds -- critique is a best-effort review,
    # not a pass/fail gate on the node's own status.
    assert result.status == "success"
    assert result.nodes[0].critique.satisfied is False
    assert result.nodes[0].critique.feedback == "still nope"
    assert result.nodes[0].response.choices[0].message["content"] == "attempt 2"
