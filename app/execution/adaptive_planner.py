"""Adaptive mid-run planning (closes the README's "multi-turn agent
loops ... mid-run" gap): after a round of DAG nodes finishes, ask the
Planner -- with the round's actual completed outputs as context, not a
hypothetical -- whether more steps are needed, and if so, append and run
ONLY those new steps against the accumulated outputs. This is the
incremental complement to app/execution/plan_runner.py's own
verify/replan loop, which is all-or-nothing (discard the whole plan,
re-run everything); this one never re-runs or discards a completed node.

Opt-in (PlanRequest.adaptive), bounded by routing.max_adaptive_rounds,
and fails open: a PlannerError or NoAvailableModelError while asking for
a continuation simply stops adding rounds and returns whatever was
accomplished so far -- never fails a request that already has real,
useful results, consistent with how every other cross-cutting XRouter
feature (quality gate, critique, verify/replan) degrades."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from app.contracts.dag import DagRunResponse
from app.contracts.planner import PlanRequest, PlanSpec
from app.contracts.response import extract_message_text
from app.core.errors import NoAvailableModelError
from app.execution.dag import DagExecutor
from app.intelligence.planner import PlannerError, generate_plan, to_dag_request
from app.observability.logging import get_logger
from app.utils.ids import new_id

if TYPE_CHECKING:
    from app.core.engine import ChatEngine

logger = get_logger("execution.adaptive_planner")

PlanMutator = Callable[[PlanSpec], PlanSpec]

_MAX_OUTPUT_SUMMARY_CHARS = 400


def _summarize_outputs(outputs: dict[str, str]) -> str:
    return "\n".join(f"- {node_id}: {text[:_MAX_OUTPUT_SUMMARY_CHARS]}" for node_id, text in outputs.items())


def _continuation_context(base_context: str | None, outputs: dict[str, str]) -> str:
    note = (
        f"Progress so far on this task:\n{_summarize_outputs(outputs)}\n\n"
        "Decide whether the task above is now fully complete. If it is, call "
        "submit_plan with an empty nodes array. If more steps are genuinely "
        "needed, call submit_plan with ONLY the new steps still required -- "
        "do not repeat steps already listed above. A new step's depends_on "
        "may reference the id of any step already completed above."
    )
    return f"{base_context}\n\n{note}" if base_context else note


def _successful_outputs(dag_result: DagRunResponse) -> dict[str, str]:
    return {r.id: extract_message_text(r.response) for r in dag_result.nodes if r.status == "success" and r.response}


async def run_adaptive_plan(
    engine: "ChatEngine", plan_request: PlanRequest, plan: PlanSpec, max_nodes: int,
    max_plan_retries: int, max_adaptive_rounds: int, plan_mutator: PlanMutator | None = None,
) -> tuple[PlanSpec, DagRunResponse, int]:
    """Runs `plan`'s nodes, then loops up to max_adaptive_rounds times
    asking the Planner for continuation nodes based on the real outputs
    accumulated so far, executing only each round's new nodes. Returns
    (the full merged plan across every round, one merged DagRunResponse,
    the number of adaptive rounds actually used)."""
    executor = DagExecutor(
        engine, max_nodes=max_nodes, tools=engine.ctx.tools,
        max_tool_iterations=engine.ctx.settings.routing.max_tool_iterations,
        max_critique_retries=engine.ctx.settings.routing.max_critique_retries,
    )

    all_nodes = list(plan.nodes)
    dag_result = await executor.run(to_dag_request(plan))
    all_results = list(dag_result.nodes)
    total_latency_ms = dag_result.latency_ms
    outputs = _successful_outputs(dag_result)

    round_count = 0
    while round_count < max_adaptive_rounds and len(all_nodes) < max_nodes:
        remaining_budget = max_nodes - len(all_nodes)
        continuation_request = plan_request.model_copy(update={
            "context": _continuation_context(plan_request.context, outputs),
        })
        try:
            continuation_plan, _ = await generate_plan(
                engine, continuation_request, remaining_budget, max_retries=max_plan_retries,
                known_ids=frozenset(outputs), allow_empty=True,
            )
        except (PlannerError, NoAvailableModelError) as e:
            logger.warning("adaptive planning round %d stopped: %s", round_count + 1, e)
            break

        if plan_mutator is not None:
            continuation_plan = plan_mutator(continuation_plan)
        if not continuation_plan.nodes:
            break

        round_count += 1
        all_nodes.extend(continuation_plan.nodes)
        round_dag_result = await executor.run(to_dag_request(continuation_plan), seed_outputs=outputs)
        all_results.extend(round_dag_result.nodes)
        total_latency_ms += round_dag_result.latency_ms
        outputs.update(_successful_outputs(round_dag_result))

    statuses = {r.status for r in all_results}
    overall = "success" if statuses == {"success"} else ("failed" if "success" not in statuses else "partial")
    merged_dag = DagRunResponse(id=new_id("dag"), status=overall, nodes=all_results, latency_ms=round(total_latency_ms, 1))
    return PlanSpec(nodes=all_nodes), merged_dag, round_count
