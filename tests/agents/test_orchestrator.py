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
from app.intelligence.counterfactual import COUNTERFACTUAL_TOOL_NAME
from app.intelligence.critic import CRITIQUE_TOOL_NAME
from app.intelligence.evidence import EVIDENCE_TOOL_NAME
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


def _evidence_tool_call(claims: list[dict]) -> dict:
    return {
        "id": "call_1", "type": "function",
        "function": {"name": EVIDENCE_TOOL_NAME, "arguments": json.dumps({"claims": claims})},
    }


def _counterfactual_tool_call(points: list[dict]) -> dict:
    return {
        "id": "call_1", "type": "function",
        "function": {"name": COUNTERFACTUAL_TOOL_NAME, "arguments": json.dumps({"points": points})},
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


# --- _recall / _remember with RetrievalConfig.enabled (EmbeddingRetriever) --

async def _engine_with_retrieval(tmp_path, embed_vectors):
    return await build_test_engine(
        tmp_path, {"solo": {"content": "ok", "embed_vectors": embed_vectors}},
        retrieval_overrides={"enabled": True, "embedding_provider": "solo", "embedding_model": "test-model"},
    )


@pytest.mark.asyncio
async def test_remember_stores_an_embedding_when_retrieval_enabled(tmp_path):
    summary = "Task: what's 2+2\nAnswer: 4"
    engine = await _engine_with_retrieval(tmp_path, {summary: [1.0, 0.0]})
    request = OrchestrationRequest(task="what's 2+2", scope="a-scope")
    await _remember(engine, request, "4")
    recent = await engine.ctx.memory_repo.recent("a-scope", limit=5)
    assert recent[0].embedding == [1.0, 0.0]


@pytest.mark.asyncio
async def test_recall_uses_embedding_retriever_when_retrieval_enabled(tmp_path):
    # _MEMORY_RECALL_TOP_K is 3, so with only two candidates both show up
    # -- what proves this is EmbeddingRetriever (cosine ranking) and not
    # KeywordRetriever (word-overlap match) is the *order*: the close
    # cosine match must be ranked ahead of the far one.
    engine = await _engine_with_retrieval(tmp_path, {"the gateway": [1.0, 0.0]})
    await engine.ctx.memory_repo.save("global", "far match", embedding=[0.0, 1.0])
    await engine.ctx.memory_repo.save("global", "close match", embedding=[1.0, 0.0])
    request = OrchestrationRequest(task="the gateway", scope="global")
    result = await _recall(engine, request)
    assert "close match" in result and "far match" in result
    assert result.index("close match") < result.index("far match")


@pytest.mark.asyncio
async def test_recall_falls_open_to_empty_when_embedding_provider_lacks_the_capability(tmp_path):
    # embed_vectors=None -> the fake never declared EMBEDDINGS, mirroring
    # an adapter that doesn't support it -- retrieve() must fail open to
    # [] rather than raise, so _recall just returns the original context.
    engine = await build_test_engine(
        tmp_path, {"solo": {"content": "ok"}},
        retrieval_overrides={"enabled": True, "embedding_provider": "solo", "embedding_model": "test-model"},
    )
    await engine.ctx.memory_repo.save("global", "some memory", embedding=[1.0, 0.0])
    request = OrchestrationRequest(task="anything", scope="global", context="original context")
    result = await _recall(engine, request)
    assert result == "original context"


@pytest.mark.asyncio
async def test_remember_saves_without_embedding_when_retrieval_disabled(tmp_path):
    # Default engine (retrieval disabled) -- regression guard that the new
    # optional embedding path never activates unless explicitly enabled.
    engine = await build_test_engine(tmp_path, {"solo": {"content": "ok", "embed_vectors": {"x": [1.0]}}})
    request = OrchestrationRequest(task="task", scope="a-scope")
    await _remember(engine, request, "answer")
    recent = await engine.ctx.memory_repo.recent("a-scope", limit=5)
    assert recent[0].embedding is None


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
async def test_orchestrate_tier4_passes_adaptive_true_tier3_passes_adaptive_false(tmp_path, monkeypatch):
    # Adaptive mid-run planning is bundled into Tier 4 ("very hard")
    # exactly like verify already is -- gated by the same verify_tier
    # boolean, never a separate opt-in field -- so Tier 3 must never see
    # PlanRequest.adaptive=True and Tier 4 must always see it True.
    import app.agents.orchestrator as orchestrator_module

    captured: list[bool] = []
    original = orchestrator_module.run_plan_with_verification

    async def _capture(engine, plan_request, **kwargs):
        captured.append(plan_request.adaptive)
        return await original(engine, plan_request, **kwargs)

    monkeypatch.setattr(orchestrator_module, "run_plan_with_verification", _capture)

    plan_json = [{"id": "solve", "prompt": "work the problem"}]
    tier3_engine = await build_test_engine(tmp_path, {"solo": {"responses": [
        {"tool_calls": [_plan_tool_call(plan_json)]},
        {"content": "an answer"},
        {"tool_calls": [_critique_tool_call(True)]},
    ]}})
    await orchestrate(tier3_engine, OrchestrationRequest(task=TASK_TIER3))

    tier4_engine = await build_test_engine(tmp_path, {"solo": {"responses": [
        {"tool_calls": [_plan_tool_call(plan_json)]},
        {"content": "an answer"},
        {"tool_calls": [_critique_tool_call(True)]},
        {"tool_calls": [_plan_tool_call([])]},          # adaptive continuation, nothing more needed
        {"tool_calls": [_verify_tool_call(True)]},
        {"content": "the case for this answer"},
        {"content": "a counterpoint to consider"},
        {"content": "the final, strengthened answer"},
    ]}})
    await orchestrate(tier4_engine, OrchestrationRequest(task=TASK_TIER4))

    assert captured == [False, True]


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
async def test_orchestrate_tier4_includes_research_verifier_and_debate(tmp_path):
    plan_json = [{"id": "solve", "prompt": "research and answer", "enable_tools": ["web_search"]}]
    engine = await build_test_engine(
        tmp_path,
        {"solo": {"responses": [
            {"tool_calls": [_plan_tool_call(plan_json)]},       # 1: plan
            {"content": "researched answer"},                    # 2: node (offered web_search, doesn't call it)
            {"tool_calls": [_critique_tool_call(True)]},          # 3: forced critique on the terminal node
            {"tool_calls": [_plan_tool_call([])]},                # 4: adaptive continuation, nothing more needed
            {"tool_calls": [_verify_tool_call(True)]},            # 5: verifier, satisfied
            {"content": "the case for this answer"},              # 6: debate advocate
            {"content": "a counterpoint to consider"},             # 7: debate skeptic
            {"content": "the final, strengthened answer"},          # 8: debate judge
        ]}},
        tool_specs={"web_search": {}},
    )
    result = await orchestrate(engine, OrchestrationRequest(task=TASK_TIER4))

    assert result.complexity == 4
    assert result.team == ["planner", "specialists", "critic", "research", "verifier", "debate"]
    # Debate's judge output must actually replace the pre-debate draft.
    assert result.answer == "the final, strengthened answer"
    assert result.verification is not None
    assert result.verification.satisfied is True
    assert result.debate is not None
    assert result.debate.position == "researched answer"
    assert result.debate.resolution == "the final, strengthened answer"
    solo = engine.ctx.providers.get("solo")
    assert solo.call_count == 8


@pytest.mark.asyncio
async def test_orchestrate_tier4_debate_fails_open_without_losing_the_pre_debate_answer(tmp_path):
    plan_json = [{"id": "solve", "prompt": "answer"}]

    def _fail_from_seventh_call(count: int) -> None:
        if count >= 7:
            from app.core.errors import ProviderServerError

            raise ProviderServerError("simulated 500", provider_id="solo")

    engine = await build_test_engine(tmp_path, {"solo": {
        "behavior": _fail_from_seventh_call,
        "responses": [
            {"tool_calls": [_plan_tool_call(plan_json)]},       # 1: plan
            {"content": "researched answer"},                    # 2: node
            {"tool_calls": [_critique_tool_call(True)]},          # 3: forced critique
            {"tool_calls": [_plan_tool_call([])]},                # 4: adaptive continuation, nothing more needed
            {"tool_calls": [_verify_tool_call(True)]},            # 5: verifier, satisfied
            {"content": "the case for this answer"},              # 6: debate advocate (succeeds)
            # call 7 (debate skeptic) fails -- the whole debate must roll back
        ],
    }})

    result = await orchestrate(engine, OrchestrationRequest(task=TASK_TIER4))

    assert result.answer == "researched answer"
    assert result.team == ["planner", "specialists", "critic", "verifier"]
    assert result.debate is None


# --- orchestrate(): trace_evidence (Phase 4: Evidence Graph) ----------------

@pytest.mark.asyncio
async def test_orchestrate_tier0_1_trace_evidence_is_a_noop(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"content": "hello there"}})
    result = await orchestrate(engine, OrchestrationRequest(task=TASK_TIER0, trace_evidence=True))
    assert result.team == ["solver"]
    assert result.evidence is None
    solo = engine.ctx.providers.get("solo")
    assert solo.call_count == 1  # no evidence-tracing call at a tier with no DAG


@pytest.mark.asyncio
async def test_orchestrate_tier2_with_trace_evidence_builds_and_attaches_evidence(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"responses": [
        {"content": "a solid answer"},
        {"tool_calls": [_critique_tool_call(True)]},
        {"tool_calls": [_evidence_tool_call([{"claim": "a solid answer", "supported_by": ["solve"], "supported": True}])]},
    ]}})
    result = await orchestrate(engine, OrchestrationRequest(task=TASK_TIER2, trace_evidence=True))

    assert result.team == ["solver", "critic", "evidence"]
    assert result.evidence is not None
    assert result.evidence.claims[0].supported_by == ["solve"]
    solo = engine.ctx.providers.get("solo")
    assert solo.call_count == 3


@pytest.mark.asyncio
async def test_orchestrate_tier3_trace_evidence_drops_a_hallucinated_node_id(tmp_path):
    plan_json = [{"id": "solve", "prompt": "work the problem"}]
    engine = await build_test_engine(tmp_path, {"solo": {"responses": [
        {"tool_calls": [_plan_tool_call(plan_json)]},
        {"content": "the final answer"},
        {"tool_calls": [_critique_tool_call(True)]},
        {"tool_calls": [_evidence_tool_call([
            {"claim": "the final answer", "supported_by": ["solve", "not_a_real_node"], "supported": True},
        ])]},
    ]}})
    result = await orchestrate(engine, OrchestrationRequest(task=TASK_TIER3, trace_evidence=True))

    assert result.team == ["planner", "specialists", "critic", "evidence"]
    assert result.evidence.claims[0].supported_by == ["solve"]


@pytest.mark.asyncio
async def test_orchestrate_trace_evidence_fails_open_without_losing_the_answer(tmp_path):
    def _fail_from_third_call(count: int) -> None:
        if count >= 3:
            from app.core.errors import ProviderServerError

            raise ProviderServerError("simulated 500", provider_id="solo")

    engine = await build_test_engine(tmp_path, {"solo": {
        "behavior": _fail_from_third_call,
        "responses": [{"content": "a solid answer"}, {"tool_calls": [_critique_tool_call(True)]}],
    }})

    result = await orchestrate(engine, OrchestrationRequest(task=TASK_TIER2, trace_evidence=True))

    # The evidence-tracing call itself failed, but that's advisory
    # metadata -- it must never take down an already-produced answer.
    assert result.answer == "a solid answer"
    assert result.team == ["solver", "critic", "evidence"]
    assert result.evidence.claims == []


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


# --- orchestrate(): trace_counterfactual (Phase 4: Counterfactual) ----------

@pytest.mark.asyncio
async def test_orchestrate_tier0_1_trace_counterfactual_works_without_a_dag(tmp_path):
    # The key behavioral difference from trace_evidence: this genuinely
    # runs at tier 0-1, since it never needs a DAG.
    engine = await build_test_engine(tmp_path, {"solo": {"responses": [
        {"content": "hello there"},
        {"tool_calls": [_counterfactual_tool_call([{"assumption": "a", "if_false": "b"}])]},
    ]}})
    result = await orchestrate(engine, OrchestrationRequest(task=TASK_TIER0, trace_counterfactual=True))

    assert result.team == ["solver", "counterfactual"]
    assert result.answer == "hello there"
    assert result.counterfactual is not None
    assert result.counterfactual.points[0].assumption == "a"
    solo = engine.ctx.providers.get("solo")
    assert solo.call_count == 2


@pytest.mark.asyncio
async def test_orchestrate_tier2_with_trace_counterfactual_attaches_analysis(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"responses": [
        {"content": "a solid answer"},
        {"tool_calls": [_critique_tool_call(True)]},
        {"tool_calls": [_counterfactual_tool_call([{"assumption": "a", "if_false": "b"}])]},
    ]}})
    result = await orchestrate(engine, OrchestrationRequest(task=TASK_TIER2, trace_counterfactual=True))

    assert result.team == ["solver", "critic", "counterfactual"]
    assert result.counterfactual is not None
    assert result.counterfactual.points[0].if_false == "b"
    solo = engine.ctx.providers.get("solo")
    assert solo.call_count == 3


@pytest.mark.asyncio
async def test_orchestrate_tier4_trace_counterfactual_runs_after_debate(tmp_path):
    plan_json = [{"id": "solve", "prompt": "work the problem"}]
    engine = await build_test_engine(tmp_path, {"solo": {"responses": [
        {"tool_calls": [_plan_tool_call(plan_json)]},           # 1: plan
        {"content": "researched answer"},                        # 2: node
        {"tool_calls": [_critique_tool_call(True)]},              # 3: forced critique
        {"tool_calls": [_plan_tool_call([])]},                    # 4: adaptive continuation, nothing more needed
        {"tool_calls": [_verify_tool_call(True)]},                # 5: verifier
        {"content": "the case for this answer"},                  # 6: debate advocate
        {"content": "a counterpoint to consider"},                # 7: debate skeptic
        {"content": "the final, strengthened answer"},            # 8: debate judge
        {"tool_calls": [_counterfactual_tool_call([{"assumption": "a", "if_false": "b"}])]},  # 9: counterfactual
    ]}})

    result = await orchestrate(engine, OrchestrationRequest(task=TASK_TIER4, trace_counterfactual=True))

    assert result.team == ["planner", "specialists", "critic", "verifier", "debate", "counterfactual"]
    assert result.answer == "the final, strengthened answer"
    assert result.counterfactual is not None
    solo = engine.ctx.providers.get("solo")
    assert solo.call_count == 9


@pytest.mark.asyncio
async def test_orchestrate_trace_counterfactual_fails_open_without_losing_the_answer(tmp_path):
    def _fail_from_third_call(count: int) -> None:
        if count >= 3:
            from app.core.errors import ProviderServerError

            raise ProviderServerError("simulated 500", provider_id="solo")

    engine = await build_test_engine(tmp_path, {"solo": {
        "behavior": _fail_from_third_call,
        "responses": [{"content": "a solid answer"}, {"tool_calls": [_critique_tool_call(True)]}],
    }})

    result = await orchestrate(engine, OrchestrationRequest(task=TASK_TIER2, trace_counterfactual=True))

    # The counterfactual call itself failed, but this is advisory
    # metadata -- it must never take down an already-produced answer.
    assert result.answer == "a solid answer"
    assert result.team == ["solver", "critic", "counterfactual"]
    assert result.counterfactual.points == []


# --- orchestrate(): simulate (Phase 4: Simulation) ---------------------------

@pytest.mark.asyncio
async def test_orchestrate_tier0_1_simulate_works_without_a_dag(tmp_path):
    # Same key behavioral point as trace_counterfactual: this genuinely
    # runs at tier 0-1, since it never needs a DAG.
    engine = await build_test_engine(tmp_path, {"solo": {"responses": [
        {"content": "hello there"},
        {"content": "under the changed premise, a different answer"},
    ]}})
    result = await orchestrate(engine, OrchestrationRequest(task=TASK_TIER0, simulate=["what if X instead"]))

    assert result.team == ["solver", "simulation"]
    assert result.answer == "hello there"
    assert len(result.simulations) == 1
    assert result.simulations[0].scenario == "what if X instead"
    assert result.simulations[0].answer == "under the changed premise, a different answer"
    solo = engine.ctx.providers.get("solo")
    assert solo.call_count == 2


@pytest.mark.asyncio
async def test_orchestrate_tier2_with_simulate_runs_two_scenarios(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"responses": [
        {"content": "a solid answer"},
        {"tool_calls": [_critique_tool_call(True)]},
        {"content": "scenario A's answer"},
        {"content": "scenario B's answer"},
    ]}})
    result = await orchestrate(engine, OrchestrationRequest(task=TASK_TIER2, simulate=["scenario A", "scenario B"]))

    assert result.team == ["solver", "critic", "simulation"]
    assert len(result.simulations) == 2
    assert result.simulations[0].answer == "scenario A's answer"
    assert result.simulations[1].answer == "scenario B's answer"
    solo = engine.ctx.providers.get("solo")
    assert solo.call_count == 4


@pytest.mark.asyncio
async def test_orchestrate_simulate_one_scenario_failing_still_keeps_the_other_and_the_answer(tmp_path):
    # A single failed candidate retries internally (RetryConfig's default
    # of 3 total attempts) before engine.handle_chat() gives up -- so the
    # "fails" scenario's request must keep failing across calls 3-5 (all
    # of its attempts), not just once, to genuinely fail rather than
    # succeed on a retry.
    def _fail_for_calls_three_through_five(count: int) -> None:
        if 3 <= count <= 5:
            from app.core.errors import ProviderServerError

            raise ProviderServerError("simulated 500", provider_id="solo")

    engine = await build_test_engine(tmp_path, {"solo": {
        "behavior": _fail_for_calls_three_through_five,
        "responses": [
            {"content": "a solid answer"},                          # 1: solver
            {"tool_calls": [_critique_tool_call(True)]},             # 2: critique
            {"content": "n/a"}, {"content": "n/a"}, {"content": "n/a"},  # 3-5: "fails" scenario's attempts
            {"content": "the surviving scenario's answer"},          # 6: "succeeds" scenario
        ],
    }})

    result = await orchestrate(engine, OrchestrationRequest(task=TASK_TIER2, simulate=["fails", "succeeds"]))

    # A failed scenario is advisory-metadata fail-open, same as evidence/
    # counterfactual -- it must never take down the already-produced answer.
    assert result.answer == "a solid answer"
    assert result.team == ["solver", "critic", "simulation"]
    assert len(result.simulations) == 1
    assert result.simulations[0].scenario == "succeeds"


@pytest.mark.asyncio
async def test_orchestrate_simulate_is_a_noop_when_not_requested(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"content": "hello there"}})
    result = await orchestrate(engine, OrchestrationRequest(task=TASK_TIER0))

    assert result.team == ["solver"]
    assert result.simulations == []
    solo = engine.ctx.providers.get("solo")
    assert solo.call_count == 1


# --- orchestrate(): Confidence Engine (Phase 4, last piece) -----------------

@pytest.mark.asyncio
async def test_orchestrate_tier0_1_confidence_is_the_unreviewed_baseline(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"content": "hello there"}})
    result = await orchestrate(engine, OrchestrationRequest(task=TASK_TIER0))

    assert result.confidence.score == 0.5
    assert result.confidence.label == "medium"
    assert result.confidence.reasons == ["unreviewed"]


@pytest.mark.asyncio
async def test_orchestrate_tier2_happy_path_confidence_is_high_with_no_reasons(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"responses": [
        {"content": "a solid answer"},
        {"tool_calls": [_critique_tool_call(True)]},
    ]}})
    result = await orchestrate(engine, OrchestrationRequest(task=TASK_TIER2))

    assert result.confidence.score == 1.0
    assert result.confidence.label == "high"
    assert result.confidence.reasons == []


@pytest.mark.asyncio
async def test_orchestrate_tier4_debate_fails_open_confidence_reflects_it(tmp_path):
    plan_json = [{"id": "solve", "prompt": "answer"}]

    def _fail_from_seventh_call(count: int) -> None:
        if count >= 7:
            from app.core.errors import ProviderServerError

            raise ProviderServerError("simulated 500", provider_id="solo")

    engine = await build_test_engine(tmp_path, {"solo": {
        "behavior": _fail_from_seventh_call,
        "responses": [
            {"tool_calls": [_plan_tool_call(plan_json)]},       # 1: plan
            {"content": "researched answer"},                    # 2: node
            {"tool_calls": [_critique_tool_call(True)]},          # 3: forced critique
            {"tool_calls": [_plan_tool_call([])]},                # 4: adaptive continuation, nothing more needed
            {"tool_calls": [_verify_tool_call(True)]},            # 5: verifier, satisfied
            {"content": "the case for this answer"},              # 6: debate advocate (succeeds)
            # call 7 (debate skeptic) fails -- the whole debate must roll back
        ],
    }})

    result = await orchestrate(engine, OrchestrationRequest(task=TASK_TIER4))

    assert result.debate is None
    assert "debate_failed_open" in result.confidence.reasons
    assert result.confidence.score < 1.0
