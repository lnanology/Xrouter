import json

import pytest

from app.contracts.planner import PlanRequest
from app.execution.plan_runner import run_plan_with_verification
from app.intelligence.planner import PLAN_TOOL_NAME
from app.intelligence.verifier import VERIFY_TOOL_NAME
from tests.helpers import build_test_engine


def _plan_tool_call(nodes: list[dict]) -> dict:
    return {"id": "call_1", "type": "function", "function": {"name": PLAN_TOOL_NAME, "arguments": json.dumps({"nodes": nodes})}}


def _verify_tool_call(satisfied: bool, feedback: str = "") -> dict:
    return {
        "id": "call_1", "type": "function",
        "function": {"name": VERIFY_TOOL_NAME, "arguments": json.dumps({"satisfied": satisfied, "feedback": feedback})},
    }


@pytest.mark.asyncio
async def test_verify_false_runs_exactly_once_with_no_verification(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"responses": [
        {"tool_calls": [_plan_tool_call([{"id": "a", "prompt": "hi"}])]},
        {"content": "ok"},
    ]}})
    plan_request = PlanRequest(task="do something", model="solo/test-model", verify=False)

    result = await run_plan_with_verification(engine, plan_request, max_nodes=20, max_plan_retries=2, max_verify_retries=1)

    assert result.verification is None
    assert result.replan_count == 0
    assert result.dag.status == "success"
    solo = engine.ctx.providers.get("solo")
    assert solo.call_count == 2  # plan + one DAG node, no verification call


@pytest.mark.asyncio
async def test_verify_true_satisfied_first_try_does_not_replan(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"responses": [
        {"tool_calls": [_plan_tool_call([{"id": "a", "prompt": "hi"}])]},
        {"content": "ok"},
        {"tool_calls": [_verify_tool_call(True)]},
    ]}})
    plan_request = PlanRequest(task="do something", model="solo/test-model", verify=True)

    result = await run_plan_with_verification(engine, plan_request, max_nodes=20, max_plan_retries=2, max_verify_retries=1)

    assert result.verification.satisfied is True
    assert result.replan_count == 0
    solo = engine.ctx.providers.get("solo")
    assert solo.call_count == 3  # plan + node + verify, no replan


@pytest.mark.asyncio
async def test_verify_true_unsatisfied_then_satisfied_replans_once(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"responses": [
        {"tool_calls": [_plan_tool_call([{"id": "a", "prompt": "draft v1"}])]},   # plan 1
        {"content": "a shallow answer"},                                          # node "a" (plan 1)
        {"tool_calls": [_verify_tool_call(False, "needs more detail")]},          # verify 1: unsatisfied
        {"tool_calls": [_plan_tool_call([{"id": "a2", "prompt": "draft v2"}])]},  # plan 2 (after feedback)
        {"content": "a thorough answer"},                                         # node "a2" (plan 2)
        {"tool_calls": [_verify_tool_call(True)]},                                # verify 2: satisfied
    ]}})
    plan_request = PlanRequest(task="write something good", model="solo/test-model", verify=True)

    result = await run_plan_with_verification(engine, plan_request, max_nodes=20, max_plan_retries=2, max_verify_retries=1)

    assert result.replan_count == 1
    assert result.verification.satisfied is True
    assert [n.id for n in result.plan.nodes] == ["a2"]  # the winning plan is the second one
    solo = engine.ctx.providers.get("solo")
    assert solo.call_count == 6
    # the 4th call is the re-plan: its request must carry the verifier's
    # feedback forward as context, proving _augment_context actually fed
    # it back in rather than just blindly retrying the identical request.
    replan_request = solo.request_log[3]
    joined = " ".join(m.content for m in replan_request.messages if isinstance(m.content, str))
    assert "needs more detail" in joined


@pytest.mark.asyncio
async def test_verify_true_gives_up_after_exhausting_retries(tmp_path):
    # Every verification comes back unsatisfied; with max_verify_retries=1
    # the loop must run exactly twice (initial + 1 replan) and then stop,
    # returning the last (still unsatisfied) result rather than looping
    # forever or erroring out.
    engine = await build_test_engine(tmp_path, {"solo": {"responses": [
        {"tool_calls": [_plan_tool_call([{"id": "a", "prompt": "v1"}])]},
        {"content": "meh"},
        {"tool_calls": [_verify_tool_call(False, "still not good enough")]},
        {"tool_calls": [_plan_tool_call([{"id": "a2", "prompt": "v2"}])]},
        {"content": "still meh"},
        {"tool_calls": [_verify_tool_call(False, "still not good enough")]},
    ]}})
    plan_request = PlanRequest(task="write something good", model="solo/test-model", verify=True)

    result = await run_plan_with_verification(engine, plan_request, max_nodes=20, max_plan_retries=2, max_verify_retries=1)

    assert result.replan_count == 1
    assert result.verification.satisfied is False
    solo = engine.ctx.providers.get("solo")
    assert solo.call_count == 6  # exactly 2 plan+execute+verify cycles, no third
