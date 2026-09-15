import json

import pytest

from app.contracts.dag import DagNodeResult, DagRunResponse
from app.contracts.response import ChatCompletionChoice, ChatCompletionResponse
from app.execution.dag_summary import summarize_dag
from app.intelligence.verifier import VERIFY_TOOL_NAME, _build_verification_request, verify
from tests.helpers import build_test_engine


def _verify_tool_call(satisfied: bool, feedback: str = "") -> dict:
    return {
        "id": "call_1", "type": "function",
        "function": {"name": VERIFY_TOOL_NAME, "arguments": json.dumps({"satisfied": satisfied, "feedback": feedback})},
    }


def _dag_result(*, success_text: dict[str, str] | None = None, failed: dict[str, str] | None = None) -> DagRunResponse:
    nodes = []
    for node_id, text in (success_text or {}).items():
        resp = ChatCompletionResponse(model="x", choices=[ChatCompletionChoice(message={"role": "assistant", "content": text})])
        nodes.append(DagNodeResult(id=node_id, status="success", response=resp))
    for node_id, err in (failed or {}).items():
        nodes.append(DagNodeResult(id=node_id, status="failed", error=err))
    statuses = {n.status for n in nodes}
    overall = "success" if statuses == {"success"} else ("failed" if "success" not in statuses else "partial")
    return DagRunResponse(id="dag_x", status=overall, nodes=nodes, latency_ms=1.0)


# --- pure -------------------------------------------------------------------

def test_summarize_dag_includes_success_output_and_failure_error():
    dag = _dag_result(success_text={"a": "Paris"}, failed={"b": "boom"})
    summary = summarize_dag(dag)
    assert "'a' (success): Paris" in summary
    assert "'b' (failed): boom" in summary


def test_summarize_dag_handles_empty_dag():
    assert summarize_dag(DagRunResponse(id="x", status="failed", nodes=[], latency_ms=0.0)) == "(no nodes ran)"


def test_build_verification_request_forces_the_submit_verification_tool():
    req = _build_verification_request("translate hello", None, _dag_result(success_text={"a": "hola"}), None)
    assert req.tool_choice == {"type": "function", "function": {"name": VERIFY_TOOL_NAME}}
    joined = " ".join(m.content for m in req.messages if isinstance(m.content, str))
    assert "translate hello" in joined
    assert "hola" in joined


# --- end-to-end via a real ChatEngine (FakeProvider-backed) ------------------

@pytest.mark.asyncio
async def test_verify_reports_satisfied(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"tool_calls": [_verify_tool_call(True)]}})
    result = await verify(engine, "task", None, _dag_result(success_text={"a": "ok"}))
    assert result.satisfied is True
    assert result.feedback is None


@pytest.mark.asyncio
async def test_verify_reports_unsatisfied_with_feedback(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"tool_calls": [_verify_tool_call(False, "missing the translation step")]}})
    result = await verify(engine, "task", None, _dag_result(success_text={"a": "ok"}))
    assert result.satisfied is False
    assert result.feedback == "missing the translation step"


@pytest.mark.asyncio
async def test_verify_fails_open_when_no_tool_call_returned(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"content": "looks fine to me"}})
    result = await verify(engine, "task", None, _dag_result(success_text={"a": "ok"}))
    assert result.satisfied is True


@pytest.mark.asyncio
async def test_verify_fails_open_on_malformed_tool_arguments(tmp_path):
    bad_call = {"id": "call_1", "type": "function", "function": {"name": VERIFY_TOOL_NAME, "arguments": "not json"}}
    engine = await build_test_engine(tmp_path, {"solo": {"tool_calls": [bad_call]}})
    result = await verify(engine, "task", None, _dag_result(success_text={"a": "ok"}))
    assert result.satisfied is True


@pytest.mark.asyncio
async def test_verify_fails_open_when_no_provider_available(tmp_path):
    engine = await build_test_engine(tmp_path, {})  # no providers registered at all
    result = await verify(engine, "task", None, _dag_result(success_text={"a": "ok"}))
    assert result.satisfied is True
