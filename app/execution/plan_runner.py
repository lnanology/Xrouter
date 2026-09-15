"""Orchestrates the full Planner -> DagExecutor -> Verifier loop (Phase 3
groundwork): generate a plan, run it, and -- if verification was
requested -- ask the Verifier whether the run actually accomplished the
task, re-planning with its feedback folded into the task context and
re-running, up to max_verify_retries times, if not. This is the one place
that wires Planner, DagExecutor and Verifier together; app/api/plan.py is
a thin HTTP wrapper around it, and tests exercise it directly against a
FakeProvider-backed engine the same way tests/execution/test_dag.py and
test_planner.py exercise their own pieces."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.contracts.dag import DagRunResponse
from app.contracts.planner import PlanRequest, PlanSpec
from app.contracts.verifier import VerificationResult
from app.execution.dag import DagExecutor
from app.intelligence.planner import generate_plan, to_dag_request
from app.intelligence.verifier import verify

if TYPE_CHECKING:
    from app.core.engine import ChatEngine


@dataclass
class PlanRunResult:
    plan: PlanSpec
    plan_attempts: int
    dag: DagRunResponse
    verification: VerificationResult | None
    replan_count: int


def _augment_context(context: str | None, feedback: str) -> str:
    note = (
        f"A previous attempt at this task was judged incomplete: {feedback}\n"
        "Produce a new plan that addresses this."
    )
    return f"{context}\n\n{note}" if context else note


async def run_plan_with_verification(
    engine: "ChatEngine", plan_request: PlanRequest, max_nodes: int,
    max_plan_retries: int, max_verify_retries: int,
) -> PlanRunResult:
    """max_verify_retries only actually bounds anything when
    plan_request.verify is True -- see PlanRequest.verify's docstring for
    why verification (and the re-planning it can trigger) is opt-in per
    request rather than a server-wide switch: it can multiply one request
    into several whole plan+execute cycles. Lets NoAvailableModelError and
    PlannerError from generate_plan(), and DagValidationError from the
    executor, propagate untouched -- the API layer (app/api/plan.py)
    turns each into the appropriate structured HTTP error."""
    current = plan_request
    verification: VerificationResult | None = None
    replan_count = 0
    retries_allowed = max_verify_retries if plan_request.verify else 0

    while True:
        plan, plan_attempts = await generate_plan(engine, current, max_nodes, max_retries=max_plan_retries)
        dag_result = await DagExecutor(
            engine, max_nodes=max_nodes,
            tools=engine.ctx.tools, max_tool_iterations=engine.ctx.settings.routing.max_tool_iterations,
        ).run(to_dag_request(plan))

        if not plan_request.verify:
            return PlanRunResult(plan=plan, plan_attempts=plan_attempts, dag=dag_result, verification=None, replan_count=0)

        verification = await verify(engine, plan_request.task, current.context, dag_result, routing_policy=current.routing_policy)
        if verification.satisfied or replan_count >= retries_allowed:
            return PlanRunResult(
                plan=plan, plan_attempts=plan_attempts, dag=dag_result,
                verification=verification, replan_count=replan_count,
            )

        replan_count += 1
        current = current.model_copy(update={
            "context": _augment_context(plan_request.context, verification.feedback or "the result did not satisfy the task"),
        })
