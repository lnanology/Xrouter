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

Debate (spec Phase 4: Evidence Graph/Debate/Counterfactual/Simulation/
Confidence Engine) is deliberately NOT part of tier 4 here -- it belongs
to the next phase, and a stub would be fake by definition. "Coder" (named
once in the spec's agents/ listing, never detailed elsewhere) also isn't
a separate stage: task_type=CODE already gets a quality-biased routing
policy from the Task Classifier, which is the existing, real behavior
this module leaves alone rather than duplicating.

Memory (app/storage/repositories/memory.py) is read before, and written
after, every tier >= 2 run: relevant past entries in the same
OrchestrationRequest.scope are retrieved and folded into the task's own
context -- a real, working retrieve-then-augment pipeline (RAG), just
keyword-based rather than semantic (see app/retrieval/'s own docstring
for why that's an honest scoping choice). Tier 0-1 stays completely
untouched by memory too, for the same "don't tax the fast path" reason."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

from app.agents.synthesizer import synthesize
from app.contracts.dag import DagNodeRequest, DagRunRequest, DagRunResponse
from app.contracts.orchestrator import OrchestrationRequest, OrchestrationResult
from app.contracts.planner import PlanRequest, PlanSpec
from app.contracts.request import ChatCompletionRequest, ChatMessage
from app.contracts.response import ChatCompletionResponse, extract_message_text
from app.core.errors import NoAvailableModelError, OrchestrationError
from app.execution.dag import DagExecutor
from app.execution.plan_runner import run_plan_with_verification
from app.intelligence.task_classifier import classify
from app.observability.logging import get_logger
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


async def _recall(engine: "ChatEngine", request: OrchestrationRequest) -> str | None:
    retriever = KeywordRetriever(engine.ctx.memory_repo)
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
    await engine.ctx.memory_repo.save(request.scope, summary)


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
        return OrchestrationResult(
            id=new_id("orch"), team=["solver"], complexity=complexity, task_type=task_type,
            answer=extract_message_text(response).strip(), latency_ms=round((time.time() - start) * 1000, 1),
        )

    context = await _recall(engine, request)

    if complexity == 2:
        dag_result = await _run_single_node_with_critique(engine, request, context)
        node = dag_result.nodes[0]
        if node.status != "success" or node.response is None:
            raise OrchestrationError(node.error or "the step failed; nothing to answer with")
        answer = extract_message_text(node.response).strip()
        await _remember(engine, request, answer)
        return OrchestrationResult(
            id=new_id("orch"), team=["solver", "critic"], complexity=complexity, task_type=task_type,
            answer=answer, dag=dag_result, latency_ms=round((time.time() - start) * 1000, 1),
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

    await _remember(engine, request, answer)

    return OrchestrationResult(
        id=new_id("orch"), team=team, complexity=complexity, task_type=task_type,
        answer=answer, dag=plan_result.dag, verification=plan_result.verification,
        latency_ms=round((time.time() - start) * 1000, 1),
    )
