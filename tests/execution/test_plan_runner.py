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


# --- adaptive (mid-run continuation rounds) -----------------------------------

@pytest.mark.asyncio
async def test_adaptive_false_never_asks_for_a_continuation(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"responses": [
        {"tool_calls": [_plan_tool_call([{"id": "a", "prompt": "hi"}])]},
        {"content": "ok"},
    ]}})
    plan_request = PlanRequest(task="do something", model="solo/test-model", adaptive=False)

    result = await run_plan_with_verification(engine, plan_request, max_nodes=20, max_plan_retries=2, max_verify_retries=1)

    assert result.adaptive_rounds == 0
    solo = engine.ctx.providers.get("solo")
    assert solo.call_count == 2  # plan + one DAG node, no continuation call at all


@pytest.mark.asyncio
async def test_adaptive_true_adds_a_node_from_a_continuation_round(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"responses": [
        {"tool_calls": [_plan_tool_call([{"id": "a", "prompt": "step a"}])]},          # initial plan
        {"content": "result a"},                                                       # node "a"
        {"tool_calls": [_plan_tool_call([{"id": "b", "prompt": "step b"}])]},          # continuation: adds "b"
        {"content": "result b"},                                                       # node "b"
        {"tool_calls": [_plan_tool_call([])]},                                         # continuation: done
    ]}})
    plan_request = PlanRequest(task="do something", model="solo/test-model", adaptive=True)

    result = await run_plan_with_verification(
        engine, plan_request, max_nodes=20, max_plan_retries=2, max_verify_retries=1, max_adaptive_rounds=3,
    )

    assert result.adaptive_rounds == 1
    assert [n.id for n in result.plan.nodes] == ["a", "b"]
    assert result.dag.status == "success"
    solo = engine.ctx.providers.get("solo")
    assert solo.call_count == 5


@pytest.mark.asyncio
async def test_adaptive_and_verify_combined_verifies_the_merged_result(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"responses": [
        {"tool_calls": [_plan_tool_call([{"id": "a", "prompt": "step a"}])]},          # initial plan
        {"content": "result a"},                                                       # node "a"
        {"tool_calls": [_plan_tool_call([])]},                                         # continuation: done, no more nodes
        {"tool_calls": [_verify_tool_call(True)]},                                     # verify: satisfied
    ]}})
    plan_request = PlanRequest(task="do something", model="solo/test-model", adaptive=True, verify=True)

    result = await run_plan_with_verification(
        engine, plan_request, max_nodes=20, max_plan_retries=2, max_verify_retries=1, max_adaptive_rounds=3,
    )

    assert result.adaptive_rounds == 0
    assert result.replan_count == 0
    assert result.verification.satisfied is True
    solo = engine.ctx.providers.get("solo")
    assert solo.call_count == 4  # plan + node + continuation(empty) + verify


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


# --- plan_mutator -------------------------------------------------------------

def _force_all_critique(plan):
    return plan.model_copy(update={"nodes": [n.model_copy(update={"critique": True}) for n in plan.nodes]})


@pytest.mark.asyncio
async def test_plan_mutator_is_applied_to_the_generated_plan_before_execution(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"responses": [
        {"tool_calls": [_plan_tool_call([{"id": "a", "prompt": "hi"}])]},
        {"content": "an answer"},
        {"tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "submit_critique", "arguments": '{"satisfied": true, "feedback": ""}'},
        }]},
    ]}})
    plan_request = PlanRequest(task="do something", model="solo/test-model", verify=False)

    result = await run_plan_with_verification(
        engine, plan_request, max_nodes=20, max_plan_retries=2, max_verify_retries=1, plan_mutator=_force_all_critique,
    )

    # The generated plan itself did not ask for critique -- only the
    # mutator did -- so this proves the mutator's output, not the
    # planner's own, is what actually got executed.
    assert result.plan.nodes[0].critique is True
    assert result.dag.nodes[0].critique is not None
    solo = engine.ctx.providers.get("solo")
    assert solo.call_count == 3  # plan + node + the critique the mutator forced


@pytest.mark.asyncio
async def test_plan_mutator_defaults_to_none_and_leaves_the_plan_untouched(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"responses": [
        {"tool_calls": [_plan_tool_call([{"id": "a", "prompt": "hi"}])]},
        {"content": "an answer"},
    ]}})
    plan_request = PlanRequest(task="do something", model="solo/test-model", verify=False)

    result = await run_plan_with_verification(engine, plan_request, max_nodes=20, max_plan_retries=2, max_verify_retries=1)

    assert result.plan.nodes[0].critique is False
    solo = engine.ctx.providers.get("solo")
    assert solo.call_count == 2  # no forced critique call


@pytest.mark.asyncio
async def test_plan_mutator_is_applied_on_every_replan(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"responses": [
        {"tool_calls": [_plan_tool_call([{"id": "a", "prompt": "v1"}])]},          # plan 1
        {"content": "shallow"},                                                     # node a (plan 1)
        {"tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "submit_critique", "arguments": '{"satisfied": true, "feedback": ""}'},
        }]},                                                                        # forced critique on node a (plan 1)
        {"tool_calls": [_verify_tool_call(False, "needs more detail")]},           # verify 1: unsatisfied
        {"tool_calls": [_plan_tool_call([{"id": "a2", "prompt": "v2"}])]},         # plan 2
        {"content": "thorough"},                                                    # node a2 (plan 2)
        {"tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "submit_critique", "arguments": '{"satisfied": true, "feedback": ""}'},
        }]},                                                                        # forced critique on node a2 (plan 2)
        {"tool_calls": [_verify_tool_call(True)]},                                  # verify 2: satisfied
    ]}})
    plan_request = PlanRequest(task="write something good", model="solo/test-model", verify=True)

    result = await run_plan_with_verification(
        engine, plan_request, max_nodes=20, max_plan_retries=2, max_verify_retries=1, plan_mutator=_force_all_critique,
    )

    assert result.replan_count == 1
    # Both the original plan's node and the re-planned node must have had
    # the mutator applied -- proving it runs on every iteration of the
    # loop, not just the first.
    assert result.plan.nodes[0].id == "a2"
    assert result.plan.nodes[0].critique is True
    solo = engine.ctx.providers.get("solo")
    assert solo.call_count == 8
