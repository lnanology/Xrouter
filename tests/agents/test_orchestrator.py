import json

import pytest

from app.agents.orchestrator import (
    _augment_context_for_tier,
    _force_terminal_critique,
    _recall,
    _remember,
    orchestrate,
)
from app.contracts.orchestrator import OrchestrationRequest
from app.contracts.planner import PlanNodeSpec, PlanSpec
from app.core.errors import OrchestrationError
from app.intelligence.critic import CRITIQUE_TOOL_NAME
from app.intelligence.planner import PLAN_TOOL_NAME
from app.intelligence.verifier import VERIFY_TOOL_NAME
from tests.helpers import build_test_engine

# Deterministic complexity-tier task texts (app.intelligence.complexity's
# HARD_KEYWORDS and length thresholds), worked out from the scoring rules
# directly rather than guessed -- see app/intelligence/complexity.py.
TASK_TIER0 = "hi"
TASK_TIER1 = "x" * 250
TASK_TIER2 = "x" * 850
TASK_TIER3 = "x" * 850 + " algorithm"
TASK_TIER4 = "x" * 850 + " algorithm optimi architecture"


def _plan_tool_call(nodes: list[dict]) -> dict:
    return {"id": "call_1", "type": "function", "function": {"name": PLAN_TOOL_NAME, "arguments": json.dumps({"nodes": nodes})}}


def _critique_tool_call(satisfied: bool, feedback: str = "") -> dict:
    return {
        "id": "call_1", "type": "function",
        "function": {"name": CRITIQUE_TOOL_NAME, "arguments": json.dumps({"satisfied": satisfied, "feedback": feedback})},
    }


def _verify_tool_call(satisfied: bool, feedback: str = "") -> dict:
    return {
        "id": "call_1", "type": "function",
        "function": {"name": VERIFY_TOOL_NAME, "arguments": json.dumps({"satisfied": satisfied, "feedback": feedback})},
    }


# --- _force_terminal_critique (pure, no engine) ------------------------------

def test_force_terminal_critique_forces_the_single_terminal_node():
    plan = PlanSpec(nodes=[PlanNodeSpec(id="a", prompt="do A")])
    mutated = _force_terminal_critique(plan)
    assert mutated.nodes[0].critique is True


def test_force_terminal_critique_leaves_depended_on_nodes_untouched():
    plan = PlanSpec(nodes=[
        PlanNodeSpec(id="a", prompt="do A"),
        PlanNodeSpec(id="b", depends_on=["a"], prompt="do B"),
    ])
    mutated = _force_terminal_critique(plan)
    by_id = {n.id: n for n in mutated.nodes}
    assert by_id["a"].critique is False  # depended-on by "b" -- not terminal
    assert by_id["b"].critique is True  # nothing depends on it -- terminal


def test_force_terminal_critique_handles_a_chain():
    plan = PlanSpec(nodes=[
        PlanNodeSpec(id="a", prompt="do A"),
        PlanNodeSpec(id="b", depends_on=["a"], prompt="do B"),
        PlanNodeSpec(id="c", depends_on=["b"], prompt="do C"),
    ])
    mutated = _force_terminal_critique(plan)
    by_id = {n.id: n for n in mutated.nodes}
    assert by_id["a"].critique is False
    assert by_id["b"].critique is False
    assert by_id["c"].critique is True


def test_force_terminal_critique_does_not_mutate_a_node_already_true():
    plan = PlanSpec(nodes=[PlanNodeSpec(id="a", prompt="do A", critique=True)])
    mutated = _force_terminal_critique(plan)
    assert mutated.nodes[0].critique is True


# --- _augment_context_for_tier (pure) ----------------------------------------

def test_augment_context_returns_context_unchanged_when_hint_not_requested():
    assert _augment_context_for_tier("existing context", hint_research=False) == "existing context"
    assert _augment_context_for_tier(None, hint_research=False) is None


def test_augment_context_appends_a_note_when_hint_requested():
    result = _augment_context_for_tier("existing context", hint_research=True)
    assert result.startswith("existing context\n\n")
    assert "enable_tools" in result


def test_augment_context_note_stands_alone_with_no_prior_context():
    result = _augment_context_for_tier(None, hint_research=True)
    assert "enable_tools" in result
    assert not result.startswith("\n\n")


# --- _recall / _remember (real MemoryRepository via a FakeProvider engine) --

@pytest.mark.asyncio
async def test_recall_returns_original_context_when_memory_is_empty(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"content": "ok"}})
    request = OrchestrationRequest(task="what's the weather", context="some context")
    result = await _recall(engine, request)
    assert result == "some context"


@pytest.mark.asyncio
async def test_recall_folds_matching_memory_into_the_context(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"content": "ok"}})
    await engine.ctx.memory_repo.save("global", "XRouter is a provider-agnostic AI gateway")
    request = OrchestrationRequest(task="tell me about the gateway", scope="global")
    result = await _recall(engine, request)
    assert "XRouter is a provider-agnostic AI gateway" in result
    assert "remembers from before" in result


@pytest.mark.asyncio
async def test_recall_is_scoped(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"content": "ok"}})
    await engine.ctx.memory_repo.save("scope-a", "the secret ingredient is basil")
    request = OrchestrationRequest(task="secret ingredient", scope="scope-b")
    result = await _recall(engine, request)
    assert result is None


@pytest.mark.asyncio
async def test_remember_saves_a_task_answer_summary(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"content": "ok"}})
    request = OrchestrationRequest(task="what's 2+2", scope="a-scope")
    await _remember(engine, request, "4")
    recent = await engine.ctx.memory_repo.recent("a-scope", limit=5)
    assert len(recent) == 1
    assert "what's 2+2" in recent[0].content
    assert "4" in recent[0].content


@pytest.mark.asyncio
async def test_remember_does_nothing_for_an_empty_answer(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"content": "ok"}})
    request = OrchestrationRequest(task="task", scope="a-scope")
    await _remember(engine, request, "")
    recent = await engine.ctx.memory_repo.recent("a-scope", limit=5)
    assert recent == []


# --- orchestrate(): tier 0-1 (trivial/simple -- solver only) ----------------

@pytest.mark.asyncio
async def test_orchestrate_tier0_uses_solver_only(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"content": "hello there"}})
    result = await orchestrate(engine, OrchestrationRequest(task=TASK_TIER0))
    assert result.complexity == 0
    assert result.team == ["solver"]
    assert result.answer == "hello there"
    assert result.dag is None
    solo = engine.ctx.providers.get("solo")
    assert solo.call_count == 1


@pytest.mark.asyncio
async def test_orchestrate_tier1_uses_solver_only(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"content": "an answer"}})
    result = await orchestrate(engine, OrchestrationRequest(task=TASK_TIER1))
    assert result.complexity == 1
    assert result.team == ["solver"]
    solo = engine.ctx.providers.get("solo")
    assert solo.call_count == 1


@pytest.mark.asyncio
async def test_orchestrate_tier0_1_never_touches_memory(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"content": "hello"}})
    await orchestrate(engine, OrchestrationRequest(task=TASK_TIER0, scope="a-scope"))
    recent = await engine.ctx.memory_repo.recent("a-scope", limit=5)
    assert recent == []


# --- orchestrate(): tier 2 (medium -- solver + critic) -----------------------

@pytest.mark.asyncio
async def test_orchestrate_tier2_runs_solver_then_critic_and_remembers(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"responses": [
        {"content": "a solid answer"},
        {"tool_calls": [_critique_tool_call(True)]},
    ]}})
    result = await orchestrate(engine, OrchestrationRequest(task=TASK_TIER2, scope="a-scope"))

    assert result.complexity == 2
    assert result.team == ["solver", "critic"]
    assert result.answer == "a solid answer"
    assert result.dag is not None
    solo = engine.ctx.providers.get("solo")
    assert solo.call_count == 2

    recent = await engine.ctx.memory_repo.recent("a-scope", limit=5)
    assert len(recent) == 1
    assert "a solid answer" in recent[0].content


@pytest.mark.asyncio
async def test_orchestrate_tier2_raises_orchestration_error_when_the_node_fails(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"behavior": "server_error"}})
    with pytest.raises(OrchestrationError):
        await orchestrate(engine, OrchestrationRequest(task=TASK_TIER2))


# --- orchestrate(): tier 3 (hard -- planner + specialists + critic) ---------

@pytest.mark.asyncio
async def test_orchestrate_tier3_single_node_forces_terminal_critique(tmp_path):
    plan_json = [{"id": "solve", "prompt": "work the problem"}]
    engine = await build_test_engine(tmp_path, {"solo": {"responses": [
        {"tool_calls": [_plan_tool_call(plan_json)]},
        {"content": "the final answer"},
        {"tool_calls": [_critique_tool_call(True)]},
    ]}})
    result = await orchestrate(engine, OrchestrationRequest(task=TASK_TIER3, scope="a-scope"))

    assert result.complexity == 3
    assert result.team == ["planner", "specialists", "critic"]
    assert result.answer == "the final answer"
    solo = engine.ctx.providers.get("solo")
    assert solo.call_count == 3  # plan + node + forced critique on the terminal node
    recent = await engine.ctx.memory_repo.recent("a-scope", limit=5)
    assert len(recent) == 1


@pytest.mark.asyncio
async def test_orchestrate_tier3_multi_node_plan_uses_synthesizer(tmp_path):
    plan_json = [
        {"id": "a", "prompt": "step one"},
        {"id": "b", "depends_on": ["a"], "prompt": "step two using {{a}}"},
    ]
    engine = await build_test_engine(tmp_path, {"solo": {"responses": [
        {"tool_calls": [_plan_tool_call(plan_json)]},  # 1: plan
        {"content": "result of step one"},               # 2: node a
        {"content": "result of step two"},                # 3: node b
        {"tool_calls": [_critique_tool_call(True)]},      # 4: forced critique on terminal node b
        {"content": "one combined final answer"},         # 5: synthesizer
    ]}})
    result = await orchestrate(engine, OrchestrationRequest(task=TASK_TIER3))

    assert result.team == ["planner", "specialists", "critic", "synthesizer"]
    assert result.answer == "one combined final answer"
    solo = engine.ctx.providers.get("solo")
    assert solo.call_count == 5


@pytest.mark.asyncio
async def test_orchestrate_tier3_all_nodes_failed_raises_orchestration_error(tmp_path):
    plan_json = [{"id": "solve", "prompt": "work the problem"}]

    def _fail_from_second_call(count: int) -> None:
        if count >= 2:
            from app.core.errors import ProviderServerError

            raise ProviderServerError("simulated 500", provider_id="solo")

    engine = await build_test_engine(tmp_path, {"solo": {
        "behavior": _fail_from_second_call,
        "responses": [{"tool_calls": [_plan_tool_call(plan_json)]}],
    }})

    with pytest.raises(OrchestrationError):
        await orchestrate(engine, OrchestrationRequest(task=TASK_TIER3))


# --- orchestrate(): tier 4 (very hard -- + research hint + verifier) -------

@pytest.mark.asyncio
async def test_orchestrate_tier4_includes_research_and_verifier_and_verifies(tmp_path):
    plan_json = [{"id": "solve", "prompt": "research and answer", "enable_tools": ["web_search"]}]
    engine = await build_test_engine(
        tmp_path,
        {"solo": {"responses": [
            {"tool_calls": [_plan_tool_call(plan_json)]},       # 1: plan
            {"content": "researched answer"},                    # 2: node (offered web_search, doesn't call it)
            {"tool_calls": [_critique_tool_call(True)]},          # 3: forced critique on the terminal node
            {"tool_calls": [_verify_tool_call(True)]},            # 4: verifier, satisfied
        ]}},
        tool_specs={"web_search": {}},
    )
    result = await orchestrate(engine, OrchestrationRequest(task=TASK_TIER4))

    assert result.complexity == 4
    assert result.team == ["planner", "specialists", "critic", "research", "verifier"]
    assert result.answer == "researched answer"
    assert result.verification is not None
    assert result.verification.satisfied is True
    solo = engine.ctx.providers.get("solo")
    assert solo.call_count == 4


@pytest.mark.asyncio
async def test_orchestrate_tier3_plan_without_tool_nodes_omits_research_tag(tmp_path):
    plan_json = [{"id": "solve", "prompt": "no tools needed here"}]
    engine = await build_test_engine(tmp_path, {"solo": {"responses": [
        {"tool_calls": [_plan_tool_call(plan_json)]},
        {"content": "an answer"},
        {"tool_calls": [_critique_tool_call(True)]},
    ]}})
    result = await orchestrate(engine, OrchestrationRequest(task=TASK_TIER3))
    assert "research" not in result.team
    assert "verifier" not in result.team
