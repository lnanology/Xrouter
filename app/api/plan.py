"""POST /v1/plan/run -- Planner (Phase 3 groundwork): turns a single
free-form task into an explicit DAG (app/intelligence/planner.py), runs
it through the exact same DagExecutor a client-supplied DAG uses, and
optionally (PlanRequest.verify) closes the loop with the Verifier
(app/intelligence/verifier.py), re-planning with its feedback if the run
didn't actually accomplish the task. All the orchestration logic lives in
app/execution/plan_runner.py; this module only translates that into
HTTP."""
from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Request

from app.contracts.planner import PlanRequest, PlanRunResponse
from app.core.errors import NoAvailableModelError
from app.execution.dag import DagValidationError
from app.execution.plan_runner import run_plan_with_verification
from app.intelligence.planner import PlannerError
from app.observability.logging import get_logger
from app.utils.ids import new_id

router = APIRouter()
logger = get_logger("api.plan")


def _error_body(message: str, type_: str = "xrouter_error", code: int = 400) -> dict:
    return {"message": message, "type": type_, "code": code}


@router.post("/v1/plan/run")
async def run_plan(request: Request, body: PlanRequest):
    ctx = request.app.state.context
    engine = request.app.state.engine
    max_nodes = body.max_nodes or ctx.settings.routing.max_dag_nodes
    if body.routing_policy is None:
        body = body.model_copy(update={"routing_policy": ctx.settings.routing.planner_routing_policy})

    start = time.time()
    try:
        result = await run_plan_with_verification(
            engine, body, max_nodes,
            max_plan_retries=ctx.settings.routing.max_plan_retries,
            max_verify_retries=ctx.settings.routing.max_verify_retries,
            max_adaptive_rounds=ctx.settings.routing.max_adaptive_rounds,
        )
    except NoAvailableModelError as e:
        # Nothing could even answer the planning call -- same "structured
        # 503, never a crash" contract as /v1/chat/completions.
        raise HTTPException(status_code=503, detail=_error_body(str(e), code=503)) from e
    except PlannerError as e:
        # Providers answered, but never with a usable plan even after
        # retries -- a client-side-fixable situation (rephrase the task,
        # raise max_nodes, ...), not a server outage.
        raise HTTPException(status_code=422, detail=_error_body(str(e), code=422)) from e
    except DagValidationError as e:
        # Should be unreachable: generate_plan() already validated the
        # exact same graph with app.execution.dag's own validator before
        # returning. Kept as a defensive net rather than trusting that two
        # separate constructions of "the same" DagRunRequest can never
        # diverge -- consistent with "never crash" over "trust it can't
        # happen".
        raise HTTPException(
            status_code=500, detail=_error_body(f"planner produced a plan that failed DAG validation unexpectedly: {e}", code=500)
        ) from e

    return PlanRunResponse(
        id=new_id("plan"), plan=result.plan, plan_attempts=result.plan_attempts, dag=result.dag,
        verification=result.verification, replan_count=result.replan_count, adaptive_rounds=result.adaptive_rounds,
        latency_ms=round((time.time() - start) * 1000, 1),
    )
