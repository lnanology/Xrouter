import json

import pytest

from app.contracts.planner import PlanNodeSpec, PlanRequest, PlanSpec
from app.execution.adaptive_planner import run_adaptive_plan
from app.intelligence.planner import PLAN_TOOL_NAME
from tests.helpers import build_test_engine


def _plan_tool_call(nodes: list[dict]) -> dict:
    return {"id": "call_1", "type": "function", "function": {"name": PLAN_TOOL_NAME, "arguments": json.dumps({"nodes": nodes})}}


@pytest.mark.asyncio
async def test_continuation_round_uses_real_prior_output_and_adds_a_node(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"responses": [
        {"content": "Paris"},                                                                                # node "a"
        {"tool_calls": [_plan_tool_call([{"id": "b", "depends_on": ["a"], "prompt": "translate {{a}} to Spanish"}])]},  # round 1: adds "b"
        {"content": "Madrid"},                                                                               # node "b"
        {"tool_calls": [_plan_tool_call([])]},                                                               # round 2: done
    ]}})
    plan_request = PlanRequest(task="capital and translation", model="solo/test-model", adaptive=True)
    initial_plan = PlanSpec(nodes=[PlanNodeSpec(id="a", prompt="what's the capital of France?")])

    plan, dag_result, rounds = await run_adaptive_plan(
        engine, plan_request, initial_plan, max_nodes=20, max_plan_retries=2, max_adaptive_rounds=3,
    )

    assert rounds == 1
    assert [n.id for n in plan.nodes] == ["a", "b"]
    assert dag_result.status == "success"
    assert [n.id for n in dag_result.nodes] == ["a", "b"]
    solo = engine.ctx.providers.get("solo")
    assert solo.call_count == 4
    # the continuation call must have carried "a"'s real output ("Paris") as context, not a hypothetical
    continuation_request = solo.request_log[1]
    joined = " ".join(m.content for m in continuation_request.messages if isinstance(m.content, str))
    assert "Paris" in joined
    # {{a}} substitution actually used "a"'s real output when node "b" ran, without re-running "a"
    assert solo.request_log[2].messages[0].content == "translate Paris to Spanish"


@pytest.mark.asyncio
async def test_immediate_empty_continuation_stops_after_zero_rounds(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"responses": [
        {"content": "an answer"},               # node "a"
        {"tool_calls": [_plan_tool_call([])]},  # continuation: nothing more needed
    ]}})
    plan_request = PlanRequest(task="do something", model="solo/test-model", adaptive=True)
    initial_plan = PlanSpec(nodes=[PlanNodeSpec(id="a", prompt="hi")])

    plan, dag_result, rounds = await run_adaptive_plan(
        engine, plan_request, initial_plan, max_nodes=20, max_plan_retries=2, max_adaptive_rounds=3,
    )

    assert rounds == 0
    assert [n.id for n in plan.nodes] == ["a"]
    assert dag_result.status == "success"
    solo = engine.ctx.providers.get("solo")
    assert solo.call_count == 2  # node "a" + exactly one continuation call, no more


@pytest.mark.asyncio
async def test_max_adaptive_rounds_caps_continuation_even_if_planner_keeps_wanting_more(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"responses": [
        {"content": "a"},                                                     # node "a"
        {"tool_calls": [_plan_tool_call([{"id": "b", "prompt": "step b"}])]},  # round 1
        {"content": "b"},                                                     # node "b"
        {"tool_calls": [_plan_tool_call([{"id": "c", "prompt": "step c"}])]},  # round 2
        {"content": "c"},                                                     # node "c"
        # max_adaptive_rounds=2 reached here -- no third continuation call should ever fire
    ]}})
    plan_request = PlanRequest(task="do something", model="solo/test-model", adaptive=True)
    initial_plan = PlanSpec(nodes=[PlanNodeSpec(id="a", prompt="step a")])

    plan, dag_result, rounds = await run_adaptive_plan(
        engine, plan_request, initial_plan, max_nodes=20, max_plan_retries=2, max_adaptive_rounds=2,
    )

    assert rounds == 2
    assert [n.id for n in plan.nodes] == ["a", "b", "c"]
    assert dag_result.status == "success"
    assert [n.id for n in dag_result.nodes] == ["a", "b", "c"]
    solo = engine.ctx.providers.get("solo")
    assert solo.call_count == 5  # a, plan-for-b, b, plan-for-c, c -- capped, no third planning call


@pytest.mark.asyncio
async def test_duplicate_id_continuation_is_rejected_and_a_retry_produces_a_new_one(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"responses": [
        {"content": "a"},                                                              # node "a"
        {"tool_calls": [_plan_tool_call([{"id": "a", "prompt": "reuses the id"}])]},    # round 1 attempt 1: rejected (dup id)
        {"tool_calls": [_plan_tool_call([{"id": "a2", "prompt": "a fresh new step"}])]},  # round 1 attempt 2: accepted
        {"content": "a2 done"},                                                         # node "a2"
        {"tool_calls": [_plan_tool_call([])]},                                          # round 2: done
    ]}})
    plan_request = PlanRequest(task="do something", model="solo/test-model", adaptive=True)
    initial_plan = PlanSpec(nodes=[PlanNodeSpec(id="a", prompt="step a")])

    plan, dag_result, rounds = await run_adaptive_plan(
        engine, plan_request, initial_plan, max_nodes=20, max_plan_retries=1, max_adaptive_rounds=3,
    )

    assert rounds == 1
    assert [n.id for n in plan.nodes] == ["a", "a2"]
    assert dag_result.status == "success"
    solo = engine.ctx.providers.get("solo")
    assert "already used" in solo.request_log[2].messages[-1].content.lower()


@pytest.mark.asyncio
async def test_planner_error_mid_round_stops_but_keeps_prior_rounds_results(tmp_path):
    # Every continuation attempt keeps reusing "a"'s id -- with
    # max_plan_retries=1, generate_plan exhausts its retries and raises
    # PlannerError, which run_adaptive_plan must catch and stop on,
    # returning what "a" already accomplished rather than failing the
    # whole request.
    engine = await build_test_engine(tmp_path, {"solo": {"responses": [
        {"content": "a"},                                                            # node "a"
        {"tool_calls": [_plan_tool_call([{"id": "a", "prompt": "duplicate 1"}])]},
        {"tool_calls": [_plan_tool_call([{"id": "a", "prompt": "duplicate 2"}])]},
    ]}})
    plan_request = PlanRequest(task="do something", model="solo/test-model", adaptive=True)
    initial_plan = PlanSpec(nodes=[PlanNodeSpec(id="a", prompt="step a")])

    plan, dag_result, rounds = await run_adaptive_plan(
        engine, plan_request, initial_plan, max_nodes=20, max_plan_retries=1, max_adaptive_rounds=3,
    )

    assert rounds == 0
    assert [n.id for n in plan.nodes] == ["a"]
    assert dag_result.status == "success"
    assert dag_result.nodes[0].response.choices[0].message["content"] == "a"
    solo = engine.ctx.providers.get("solo")
    assert solo.call_count == 3  # node "a" + 2 rejected planning attempts, then gave up


@pytest.mark.asyncio
async def test_no_available_model_error_mid_round_stops_but_keeps_prior_rounds_results(tmp_path):
    def _fail_from_second_call(call_count):
        if call_count >= 2:
            from app.core.errors import ProviderServerError
            raise ProviderServerError("simulated 500", provider_id="solo")

    engine = await build_test_engine(tmp_path, {"solo": {"behavior": _fail_from_second_call, "content": "a"}})
    plan_request = PlanRequest(task="do something", model="solo/test-model", adaptive=True)
    initial_plan = PlanSpec(nodes=[PlanNodeSpec(id="a", prompt="step a")])

    plan, dag_result, rounds = await run_adaptive_plan(
        engine, plan_request, initial_plan, max_nodes=20, max_plan_retries=2, max_adaptive_rounds=3,
    )

    assert rounds == 0
    assert [n.id for n in plan.nodes] == ["a"]
    assert dag_result.status == "success"
    assert dag_result.nodes[0].response.choices[0].message["content"] == "a"


@pytest.mark.asyncio
async def test_plan_mutator_is_applied_to_each_continuation_round(tmp_path):
    def _force_all_critique(plan):
        return plan.model_copy(update={"nodes": [n.model_copy(update={"critique": True}) for n in plan.nodes]})

    engine = await build_test_engine(tmp_path, {"solo": {"responses": [
        {"content": "a"},                                                     # node "a"
        {"tool_calls": [_plan_tool_call([{"id": "b", "prompt": "step b"}])]},  # round 1 (mutator not asked about "a" here -- run_adaptive_plan doesn't re-mutate the already-run initial plan, only its own continuation rounds)
        {"content": "b"},                                                     # node "b"
        {"tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "submit_critique", "arguments": '{"satisfied": true, "feedback": ""}'},
        }]},                                                                   # forced critique on "b" (mutator applied)
        {"tool_calls": [_plan_tool_call([])]},                                 # round 2: done
    ]}})
    plan_request = PlanRequest(task="do something", model="solo/test-model", adaptive=True)
    initial_plan = PlanSpec(nodes=[PlanNodeSpec(id="a", prompt="step a")])

    plan, dag_result, rounds = await run_adaptive_plan(
        engine, plan_request, initial_plan, max_nodes=20, max_plan_retries=2, max_adaptive_rounds=3,
        plan_mutator=_force_all_critique,
    )

    assert rounds == 1
    assert [n.id for n in plan.nodes] == ["a", "b"]
    assert plan.nodes[1].critique is True
    assert dag_result.nodes[1].critique is not None
