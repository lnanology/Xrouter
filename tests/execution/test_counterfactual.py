import json

import pytest

from app.contracts.dag import DagNodeResult, DagRunResponse
from app.contracts.response import ChatCompletionChoice, ChatCompletionResponse
from app.intelligence.counterfactual import (
    COUNTERFACTUAL_TOOL_NAME,
    _build_counterfactual_request,
    _sanitize_points,
    build_counterfactual_analysis,
)
from tests.helpers import build_test_engine


def _counterfactual_tool_call(points: list[dict]) -> dict:
    return {
        "id": "call_1", "type": "function",
        "function": {"name": COUNTERFACTUAL_TOOL_NAME, "arguments": json.dumps({"points": points})},
    }


def _node(id_: str, content: str = "x") -> DagNodeResult:
    response = ChatCompletionResponse(model="x", choices=[ChatCompletionChoice(message={"role": "assistant", "content": content})])
    return DagNodeResult(id=id_, status="success", response=response)


def _dag(nodes: list[DagNodeResult]) -> DagRunResponse:
    return DagRunResponse(id="dag_test", status="success", nodes=nodes, latency_ms=0.0)


# --- pure request-building -----------------------------------------------------

def test_build_counterfactual_request_forces_the_submit_tool():
    req = _build_counterfactual_request("task", "answer", None, None, None)
    assert req.tool_choice == {"type": "function", "function": {"name": COUNTERFACTUAL_TOOL_NAME}}
    joined = " ".join(m.content for m in req.messages if isinstance(m.content, str))
    assert "task" in joined
    assert "answer" in joined


def test_build_counterfactual_request_works_without_a_dag():
    req = _build_counterfactual_request("task", "answer", None, None, None)
    joined = " ".join(m.content for m in req.messages if isinstance(m.content, str))
    assert "What each step produced" not in joined


def test_build_counterfactual_request_includes_dag_summary_when_given():
    dag = _dag([_node("a", "step output")])
    req = _build_counterfactual_request("task", "answer", None, dag, None)
    joined = " ".join(m.content for m in req.messages if isinstance(m.content, str))
    assert "step output" in joined


def test_build_counterfactual_request_includes_context_when_given():
    req = _build_counterfactual_request("task", "answer", "the user is a beginner", None, None)
    joined = " ".join(m.content for m in req.messages if isinstance(m.content, str))
    assert "the user is a beginner" in joined


# --- _sanitize_points (pure) ----------------------------------------------------

def test_sanitize_points_keeps_well_formed_entries():
    points = _sanitize_points([{"assumption": "a", "if_false": "b"}])
    assert points[0].assumption == "a"
    assert points[0].if_false == "b"


def test_sanitize_points_skips_entries_missing_assumption():
    points = _sanitize_points([{"if_false": "b"}])
    assert points == []


def test_sanitize_points_skips_entries_missing_if_false():
    points = _sanitize_points([{"assumption": "a"}])
    assert points == []


# --- end-to-end build_counterfactual_analysis() via a real ChatEngine ----------

@pytest.mark.asyncio
async def test_build_counterfactual_analysis_with_a_dag(tmp_path):
    dag = _dag([_node("capital")])
    engine = await build_test_engine(tmp_path, {"solo": {"tool_calls": [_counterfactual_tool_call([
        {"assumption": "the question refers to mainland France", "if_false": "the capital would likely be different for an overseas territory"},
    ])]}})

    analysis = await build_counterfactual_analysis(engine, "what's the capital of France?", "Paris", dag=dag)

    assert len(analysis.points) == 1
    assert analysis.points[0].assumption == "the question refers to mainland France"


@pytest.mark.asyncio
async def test_build_counterfactual_analysis_without_a_dag(tmp_path):
    # No dag param at all -- the tier-0-1 shape, where no DAG ever ran.
    engine = await build_test_engine(tmp_path, {"solo": {"tool_calls": [_counterfactual_tool_call([
        {"assumption": "some assumption", "if_false": "some consequence"},
    ])]}})

    analysis = await build_counterfactual_analysis(engine, "task", "answer")

    assert len(analysis.points) == 1


@pytest.mark.asyncio
async def test_build_counterfactual_analysis_fails_open_when_no_provider_available(tmp_path):
    engine = await build_test_engine(tmp_path, {})  # no providers registered

    analysis = await build_counterfactual_analysis(engine, "task", "answer")

    assert analysis.points == []


@pytest.mark.asyncio
async def test_build_counterfactual_analysis_fails_open_when_no_tool_call_returned(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"content": "I'll just answer directly"}})

    analysis = await build_counterfactual_analysis(engine, "task", "answer")

    assert analysis.points == []


@pytest.mark.asyncio
async def test_build_counterfactual_analysis_fails_open_on_malformed_tool_arguments(tmp_path):
    bad_call = {"id": "call_1", "type": "function", "function": {"name": COUNTERFACTUAL_TOOL_NAME, "arguments": "not json"}}
    engine = await build_test_engine(tmp_path, {"solo": {"tool_calls": [bad_call]}})

    analysis = await build_counterfactual_analysis(engine, "task", "answer")

    assert analysis.points == []


@pytest.mark.asyncio
async def test_build_counterfactual_analysis_can_return_an_empty_list(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"tool_calls": [_counterfactual_tool_call([])]}})

    analysis = await build_counterfactual_analysis(engine, "task", "answer")

    assert analysis.points == []
