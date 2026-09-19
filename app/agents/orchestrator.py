"""Orchestrator (Phase 3: Dynamic Agent Team, spec section 十九): the
piece app/intelligence/complexity.py's own module docstring flagged as
"not yet built" -- it turns a single free-form task straight into
whichever team of the pieces already built (plain chat, the Critic, the
Planner + DagExecutor, the Researcher's web_search, the Verifier, the
Synthesizer) the task's own complexity (app/intelligence/
task_classifier.py, 0..4) actually calls for, following the tier table
from spec section 七:

  0-1 (trivial/simple): a single fast model answers directly -- the
      existing Phase-1 fast path, completely unchanged, zero extra calls,
      zero memory overhead. Spec section 六: "不要所有 request 都進 Agent."
  2   (medium): a single model answers, then the Critic reviews it and
      redoes it if unsatisfied (one DagExecutor node, critique=true).
  3   (hard): the Planner designs a multi-step DAG ("specialists"); every
      terminal node (nothing else depends on it) is forced to
      critique=true, deterministically, rather than trusting the Planner
      to remember -- see _force_terminal_critique. Whenever more than one
      node ran, the Synthesizer composes the final answer.
  4   (very hard): same as tier 3, plus a mandatory Verifier pass
      (reusing app/execution/plan_runner.py's own bounded
      re-plan-on-failure loop wholesale, via its verify=true path) and a
      stronger hint in the Planner's own context that a step may
      genuinely need Research (enable_tools) -- never forced, since
      forcing a tool call the task doesn't actually need would be exactly
      the "fake placeholder functionality" XRouter's rules forbid.

Evidence Graph (Phase 4's first piece, app/intelligence/evidence.py) is
opt-in via OrchestrationRequest.trace_evidence: when a DAG actually ran
(tier >= 2) it traces the final answer's own claims back to whichever
step produced each one, or flags a claim as untraceable ("model
knowledge, not checked"). Debate (Phase 4's second piece,
app/agents/debate.py) is, unlike Evidence Graph, a standing member of the
tier-4 team per spec section 十九's own "Very hard" table -- no opt-in
flag, it just runs: an Advocate and a Skeptic argue for and against the
tier's draft answer, and a Judge reconciles both into a final,
strengthened answer that replaces the draft, failing open to the
untouched draft on any problem. Counterfactual (Phase 4's third piece,
app/intelligence/counterfactual.py) is, like Evidence Graph, opt-in
(OrchestrationRequest.trace_counterfactual) rather than a spec-named
standing team member -- but unlike Evidence Graph, it runs at every
tier including 0-1, since identifying the final answer's own load-bearing
assumptions needs only the task and the answer, never a DAG. It's pure
advisory annotation, same as Evidence Graph -- it never revises the
answer itself, that's Debate's job. Simulation (Phase 4's fourth piece,
app/agents/simulation.py) is also opt-in (OrchestrationRequest.simulate,
a caller-supplied list of changed-premise scenarios), and also runs at
every tier including 0-1 for the same "doesn't need a DAG" reason -- but
unlike Counterfactual, which guesses how the answer would change,
Simulation actually re-answers the task once per scenario, a genuine
re-execution rather than a judgment. Confidence Engine (Phase 4's fifth
and last piece, app/intelligence/confidence.py) closes Phase 4 out: unlike
every piece above, it is neither opt-in nor a standing team member with
its own team-list entry -- it makes zero provider calls, so it just runs
automatically on every OrchestrationResult, at every tier, rolling up
whichever of dag/verification/debate/evidence signals this run actually
produced into one deterministic confidence score. "Coder" (named
once in the spec's agents/ listing, never detailed elsewhere) also isn't
a separate stage: task_type=CODE already gets a quality-biased routing
policy from the Task Classifier, which is the existing, real behavior
this module leaves alone rather than duplicating.

Memory (app/storage/repositories/memory.py) is read before, and written
after, every tier >= 2 run: relevant past entries in the same
OrchestrationRequest.scope are retrieved and folded into the task's own
context -- a real, working retrieve-then-augment pipeline (RAG).
_build_retriever() picks which Retriever implementation does that:
keyword/substring search by default (free, always available), or, when
RetrievalConfig.enabled, app/retrieval/embedding.py's EmbeddingRetriever
-- a real embedding + cosine-similarity search, which also makes
_remember() embed the new summary at save time. See app/retrieval/'s own
docstring for both implementations. Tier 0-1 stays completely untouched
by memory too, for the same "don't tax the fast path" reason."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

from app.agents.debate import debate
from app.agents.simulation import run_simulations
from app.agents.synthesizer import synthesize
from app.contracts.confidence import ConfidenceAssessment
from app.contracts.counterfactual import CounterfactualAnalysis
from app.contracts.dag import DagNodeRequest, DagRunRequest, DagRunResponse
from app.contracts.debate import DebateResult
from app.contracts.evidence import EvidenceGraph
from app.contracts.orchestrator import OrchestrationRequest, OrchestrationResult
from app.contracts.planner import PlanRequest, PlanSpec
from app.contracts.request import ChatCompletionRequest, ChatMessage
from app.contracts.response import ChatCompletionResponse, extract_message_text
from app.contracts.simulation import SimulationRun
from app.core.errors import NoAvailableModelError, OrchestrationError
from app.execution.dag import DagExecutor
from app.execution.plan_runner import run_plan_with_verification
from app.intelligence.confidence import assess_confidence
from app.intelligence.counterfactual import build_counterfactual_analysis
from app.intelligence.evidence import build_evidence_graph
from app.intelligence.task_classifier import classify
from app.observability.logging import get_logger
from app.retrieval.base import Retriever
from app.retrieval.embedding import EmbeddingRetriever, embed_for_memory
from app.retrieval.keyword import KeywordRetriever
from app.utils.ids import new_id

if TYPE_CHECKING:
    from app.core.engine import ChatEngine

logger = get_logger("agents.orchestrator")

_MEMORY_RECALL_TOP_K = 3
_MEMORY_SUMMARY_MAX_CHARS = 1000


def _force_terminal_critique(plan: PlanSpec) -> PlanSpec:
    """Deterministically guarantees the Critic actually reviews whichever
    node(s) a plan's own final output depends on -- the terminal,
    no-dependents nodes -- rather than trusting the Planner to have
    remembered to set critique itself. Cheap (usually exactly one node)
    and never wrong: a terminal node's own output IS the thing whose
    correctness matters most, by definition of being terminal."""
    depended_on = {dep for n in plan.nodes for dep in n.depends_on}
    nodes = [n if n.id in depended_on else n.model_copy(update={"critique": True}) for n in plan.nodes]
    return plan.model_copy(update={"nodes": nodes})


def _augment_context_for_tier(context: str | None, hint_research: bool) -> str | None:
    if not hint_research:
        return context
    note = (
        "This is a high-complexity task -- if any step genuinely needs "
        "current or external information you can't be confident of from "
        "your own knowledge, use enable_tools for that step. Only do this "
        "when a step actually needs it."
    )
    return f"{context}\n\n{note}" if context else note


def _build_retriever(engine: "ChatEngine") -> Retriever:
    """Picks the Retriever implementation from config, once per call --
    EmbeddingRetriever (real vector similarity, costs a provider call)
    when RetrievalConfig.enabled, else the always-on, zero-extra-cost
    KeywordRetriever. See app/retrieval/base.py's module docstring."""
    retrieval = engine.ctx.settings.retrieval
    if retrieval.enabled:
        return EmbeddingRetriever(
            engine.ctx.memory_repo, engine.ctx.providers,
            retrieval.embedding_provider, retrieval.embedding_model,
            max_candidates=retrieval.max_candidates,
        )
    return KeywordRetriever(engine.ctx.memory_repo)


async def _recall(engine: "ChatEngine", request: OrchestrationRequest) -> str | None:
    retriever = _build_retriever(engine)
    chunks = await retriever.retrieve(request.scope, request.task, top_k=_MEMORY_RECALL_TOP_K)
    if not chunks:
        return request.context
    recalled = "\n".join(f"- {c.content}" for c in chunks)
    note = f"Relevant things XRouter remembers from before:\n{recalled}"
    return f"{request.context}\n\n{note}" if request.context else note


async def _remember(engine: "ChatEngine", request: OrchestrationRequest, answer: str) -> None:
    if not answer:
        return
    summary = f"Task: {request.task}\nAnswer: {answer}"[:_MEMORY_SUMMARY_MAX_CHARS]
    retrieval = engine.ctx.settings.retrieval
    embedding = None
    if retrieval.enabled:
        embedding = await embed_for_memory(
            engine.ctx.providers, retrieval.embedding_provider, retrieval.embedding_model, summary,
        )
    await engine.ctx.memory_repo.save(request.scope, summary, embedding=embedding)


async def _run_solver(engine: "ChatEngine", request: OrchestrationRequest) -> ChatCompletionResponse:
    messages = []
    if request.context:
        messages.append(ChatMessage(role="system", content=f"Context:\n{request.context}"))
    messages.append(ChatMessage(role="user", content=request.task))
    return await engine.handle_chat(ChatCompletionRequest(model=request.model, routing_policy=request.routing_policy, messages=messages))


async def _run_single_node_with_critique(engine: "ChatEngine", request: OrchestrationRequest, context: str | None) -> DagRunResponse:
    prompt = f"Context:\n{context}\n\n{request.task}" if context else request.task
    node = DagNodeRequest(
        id="solve", messages=[ChatMessage(role="user", content=prompt)],
        model=request.model, routing_policy=request.routing_policy, critique=True,
    )
    routing = engine.ctx.settings.routing
    return await DagExecutor(
        engine, max_nodes=1, tools=engine.ctx.tools,
        max_tool_iterations=routing.max_tool_iterations, max_critique_retries=routing.max_critique_retries,
    ).run(DagRunRequest(nodes=[node]))


async def _maybe_trace_evidence(
    engine: "ChatEngine", request: OrchestrationRequest, answer: str, dag: DagRunResponse, team: list[str],
) -> EvidenceGraph | None:
    """Builds an Evidence Graph for the finished answer when
    trace_evidence was requested, and records "evidence" in `team` when it
    actually ran -- mutates `team` in place, mirroring how the "research"/
    "synthesizer"/"verifier" tags are only ever added once something
    actually happened, never because a tier "should" have used it."""
    if not request.trace_evidence:
        return None
    evidence = await build_evidence_graph(engine, request.task, answer, dag, routing_policy=request.routing_policy)
    team.append("evidence")
    return evidence


async def _maybe_trace_counterfactual(
    engine: "ChatEngine", request: OrchestrationRequest, answer: str, dag: DagRunResponse | None, team: list[str],
) -> CounterfactualAnalysis | None:
    """Builds a Counterfactual analysis for the finished answer when
    trace_counterfactual was requested. Unlike _maybe_trace_evidence,
    `dag` may genuinely be None here -- identifying an answer's own
    load-bearing assumptions never requires a DAG, so this runs at every
    complexity tier, including 0-1, not just tier >= 2."""
    if not request.trace_counterfactual:
        return None
    analysis = await build_counterfactual_analysis(
        engine, request.task, answer, context=request.context, dag=dag, routing_policy=request.routing_policy,
    )
    team.append("counterfactual")
    return analysis


async def _maybe_run_simulations(
    engine: "ChatEngine", request: OrchestrationRequest, team: list[str],
) -> list[SimulationRun]:
    """Actually re-answers the task once per caller-supplied scenario in
    request.simulate -- a genuine re-execution, not a guess (that's
    _maybe_trace_counterfactual's job). Unlike the evidence/counterfactual
    helpers, "simulation" is only added to `team` when at least one
    scenario actually produced a result: there's no honest "attempted but
    empty" state here, an entirely-failed batch just didn't run."""
    if not request.simulate:
        return []
    runs = await run_simulations(
        engine, request.task, request.simulate, context=request.context, routing_policy=request.routing_policy,
    )
    if runs:
        team.append("simulation")
    return runs


async def _synthesize_or_fallback(engine: "ChatEngine", request: OrchestrationRequest, dag: DagRunResponse) -> tuple[str, bool]:
    """Returns (answer, used_synthesizer). Raises OrchestrationError if
    every node in the DAG failed -- there is nothing honest to synthesize
    or fall back to."""
    successful = [n for n in dag.nodes if n.status == "success" and n.response is not None]
    if not successful:
        raise OrchestrationError("every step in the plan failed; nothing to synthesize an answer from")
    if len(successful) == 1:
        return extract_message_text(successful[0].response).strip(), False

    try:
        response = await synthesize(engine, request.task, request.context, dag, routing_policy=request.routing_policy)
    except NoAvailableModelError as e:
        logger.warning("synthesizer could not get an answer (%s); falling back to the DAG's own terminal output(s)", e)
        return "\n\n".join(extract_message_text(n.response).strip() for n in successful), False
    return extract_message_text(response).strip(), True


async def orchestrate(engine: "ChatEngine", request: OrchestrationRequest) -> OrchestrationResult:
    start = time.time()
    classification = classify(ChatCompletionRequest(model=request.model, messages=[ChatMessage(role="user", content=request.task)]))
    complexity = classification.complexity
    task_type = classification.task_type.value

    if complexity <= 1:
        response = await _run_solver(engine, request)
        answer = extract_message_text(response).strip()
        team = ["solver"]
        counterfactual = await _maybe_trace_counterfactual(engine, request, answer, None, team)
        simulations = await _maybe_run_simulations(engine, request, team)
        confidence = assess_confidence(team)
        return OrchestrationResult(
            id=new_id("orch"), team=team, complexity=complexity, task_type=task_type,
            answer=answer, counterfactual=counterfactual, simulations=simulations, confidence=confidence,
            latency_ms=round((time.time() - start) * 1000, 1),
        )

    context = await _recall(engine, request)

    if complexity == 2:
        dag_result = await _run_single_node_with_critique(engine, request, context)
        node = dag_result.nodes[0]
        if node.status != "success" or node.response is None:
            raise OrchestrationError(node.error or "the step failed; nothing to answer with")
        answer = extract_message_text(node.response).strip()
        team = ["solver", "critic"]
        evidence = await _maybe_trace_evidence(engine, request, answer, dag_result, team)
        counterfactual = await _maybe_trace_counterfactual(engine, request, answer, dag_result, team)
        simulations = await _maybe_run_simulations(engine, request, team)
        await _remember(engine, request, answer)
        confidence = assess_confidence(team, dag=dag_result)
        return OrchestrationResult(
            id=new_id("orch"), team=team, complexity=complexity, task_type=task_type,
            answer=answer, dag=dag_result, evidence=evidence, counterfactual=counterfactual, simulations=simulations,
            confidence=confidence, latency_ms=round((time.time() - start) * 1000, 1),
        )

    verify_tier = complexity >= 4
    routing = engine.ctx.settings.routing
    plan_request = PlanRequest(
        task=request.task, context=_augment_context_for_tier(context, hint_research=verify_tier),
        model=request.model, routing_policy=request.routing_policy, verify=verify_tier,
    )
    plan_result = await run_plan_with_verification(
        engine, plan_request, max_nodes=routing.max_dag_nodes,
        max_plan_retries=routing.max_plan_retries, max_verify_retries=routing.max_verify_retries,
        plan_mutator=_force_terminal_critique,
    )

    team = ["planner", "specialists", "critic"]  # critique is always forced onto at least the terminal node(s)
    if any(n.enable_tools for n in plan_result.plan.nodes):
        team.append("research")
    if verify_tier:
        team.append("verifier")

    answer, used_synthesizer = await _synthesize_or_fallback(engine, request, plan_result.dag)
    if used_synthesizer:
        team.append("synthesizer")

    debate_result: DebateResult | None = None
    if verify_tier:  # Debate is a standing tier-4 team member, spec section 十九
        answer, debate_result = await debate(
            engine, request.task, request.context, plan_result.dag, answer, routing_policy=request.routing_policy,
        )
        if debate_result is not None:
            team.append("debate")

    evidence = await _maybe_trace_evidence(engine, request, answer, plan_result.dag, team)
    counterfactual = await _maybe_trace_counterfactual(engine, request, answer, plan_result.dag, team)
    simulations = await _maybe_run_simulations(engine, request, team)
    await _remember(engine, request, answer)
    confidence = assess_confidence(
        team, dag=plan_result.dag, verification=plan_result.verification, debate=debate_result, evidence=evidence,
    )

    return OrchestrationResult(
        id=new_id("orch"), team=team, complexity=complexity, task_type=task_type,
        answer=answer, dag=plan_result.dag, verification=plan_result.verification, debate=debate_result,
        evidence=evidence, counterfactual=counterfactual, simulations=simulations,
        confidence=confidence, latency_ms=round((time.time() - start) * 1000, 1),
    )
