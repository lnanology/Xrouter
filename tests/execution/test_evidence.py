import json

import pytest

from app.contracts.dag import DagNodeResult, DagRunResponse
from app.contracts.response import ChatCompletionChoice, ChatCompletionResponse
from app.intelligence.evidence import (
    EVIDENCE_TOOL_NAME,
    _build_evidence_request,
    _build_evidence_tool_schema,
    _sanitize_claims,
    build_evidence_graph,
)
from tests.helpers import build_test_engine


def _evidence_tool_call(claims: list[dict]) -> dict:
    return {
        "id": "call_1", "type": "function",
        "function": {"name": EVIDENCE_TOOL_NAME, "arguments": json.dumps({"claims": claims})},
    }


def _node(id_: str, status: str = "success", content: str = "x") -> DagNodeResult:
    response = ChatCompletionResponse(model="x", choices=[ChatCompletionChoice(message={"role": "assistant", "content": content})])
    return DagNodeResult(id=id_, status=status, response=response)


def _dag(nodes: list[DagNodeResult]) -> DagRunResponse:
    return DagRunResponse(id="dag_test", status="success", nodes=nodes, latency_ms=0.0)


# --- pure request-building / schema -------------------------------------------

def test_build_evidence_tool_schema_enum_lists_node_ids_plus_model_knowledge():
    schema = _build_evidence_tool_schema(["a", "b"])
    items_schema = schema["function"]["parameters"]["properties"]["claims"]["items"]["properties"]["supported_by"]["items"]
    assert items_schema["enum"] == ["a", "b", "model_knowledge"]


def test_build_evidence_request_forces_the_submit_evidence_graph_tool():
    dag = _dag([_node("a")])
    req = _build_evidence_request("task", "answer", dag, None)
    assert req.tool_choice == {"type": "function", "function": {"name": EVIDENCE_TOOL_NAME}}
    joined = " ".join(m.content for m in req.messages if isinstance(m.content, str))
    assert "task" in joined
    assert "answer" in joined


# --- _sanitize_claims (pure) ---------------------------------------------------

def test_sanitize_claims_keeps_valid_ids():
    claims = _sanitize_claims([{"claim": "x", "supported_by": ["a"], "supported": True}], valid_ids={"a", "model_knowledge"})
    assert claims[0].supported_by == ["a"]


def test_sanitize_claims_drops_hallucinated_ids():
    claims = _sanitize_claims([{"claim": "x", "supported_by": ["a", "fake_node"], "supported": True}], valid_ids={"a", "model_knowledge"})
    assert claims[0].supported_by == ["a"]


def test_sanitize_claims_skips_entries_missing_claim_text():
    claims = _sanitize_claims([{"supported_by": ["a"], "supported": True}], valid_ids={"a"})
    assert claims == []


def test_sanitize_claims_defaults_supported_true_when_omitted():
    claims = _sanitize_claims([{"claim": "x", "supported_by": []}], valid_ids={"a"})
    assert claims[0].supported is True


# --- end-to-end build_evidence_graph() via a real ChatEngine ------------------

@pytest.mark.asyncio
async def test_build_evidence_graph_maps_claims_to_nodes(tmp_path):
    dag = _dag([_node("capital"), _node("translate")])
    engine = await build_test_engine(tmp_path, {"solo": {"tool_calls": [_evidence_tool_call([
        {"claim": "Paris is the capital of France", "supported_by": ["capital"], "supported": True},
        {"claim": "Madrid is a lovely city", "supported_by": [], "supported": False},
    ])]}})

    graph = await build_evidence_graph(engine, "task", "answer", dag)

    assert len(graph.claims) == 2
    assert graph.claims[0].claim == "Paris is the capital of France"
    assert graph.claims[0].supported_by == ["capital"]
    assert graph.claims[1].supported is False


@pytest.mark.asyncio
async def test_build_evidence_graph_drops_a_hallucinated_node_reference(tmp_path):
    dag = _dag([_node("a")])
    engine = await build_test_engine(tmp_path, {"solo": {"tool_calls": [_evidence_tool_call([
        {"claim": "some claim", "supported_by": ["a", "not_a_real_node"], "supported": True},
    ])]}})

    graph = await build_evidence_graph(engine, "task", "answer", dag)

    assert graph.claims[0].supported_by == ["a"]


@pytest.mark.asyncio
async def test_build_evidence_graph_fails_open_when_no_provider_available(tmp_path):
    dag = _dag([_node("a")])
    engine = await build_test_engine(tmp_path, {})  # no providers registered

    graph = await build_evidence_graph(engine, "task", "answer", dag)

    assert graph.claims == []


@pytest.mark.asyncio
async def test_build_evidence_graph_fails_open_when_no_tool_call_returned(tmp_path):
    dag = _dag([_node("a")])
    engine = await build_test_engine(tmp_path, {"solo": {"content": "I'll just answer directly"}})

    graph = await build_evidence_graph(engine, "task", "answer", dag)

    assert graph.claims == []


@pytest.mark.asyncio
async def test_build_evidence_graph_fails_open_on_malformed_tool_arguments(tmp_path):
    dag = _dag([_node("a")])
    bad_call = {"id": "call_1", "type": "function", "function": {"name": EVIDENCE_TOOL_NAME, "arguments": "not json"}}
    engine = await build_test_engine(tmp_path, {"solo": {"tool_calls": [bad_call]}})

    graph = await build_evidence_graph(engine, "task", "answer", dag)

    assert graph.claims == []
