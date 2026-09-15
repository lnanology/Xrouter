import pytest

from app.contracts.dag import DagNodeRequest, DagRunRequest
from app.execution.dag import DagExecutor, DagValidationError, _substitute, _validate_and_order
from tests.helpers import build_test_engine


def _node(id_, depends_on=None, content="hi", model="auto"):
    return DagNodeRequest(id=id_, depends_on=depends_on or [], messages=[{"role": "user", "content": content}], model=model)


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
