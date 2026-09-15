"""POST /v1/agents/run -- Dynamic Agent Team (Phase 3, spec section 十九):
turns a single free-form task into whichever team (app/agents/
orchestrator.py) the task's own complexity actually calls for, and
returns one final answer. All the team-assembly logic lives in
app/agents/orchestrator.py; this module only translates that into HTTP,
the same thin-wrapper pattern app/api/dag.py and app/api/plan.py follow."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.agents.orchestrator import orchestrate
from app.contracts.orchestrator import OrchestrationRequest, OrchestrationResult
from app.core.errors import NoAvailableModelError, OrchestrationError
from app.execution.dag import DagValidationError
from app.intelligence.planner import PlannerError
from app.observability.logging import get_logger

router = APIRouter()
logger = get_logger("api.orchestrator")


def _error_body(message: str, type_: str = "xrouter_error", code: int = 400) -> dict:
    return {"message": message, "type": type_, "code": code}


@router.post("/v1/agents/run", response_model=OrchestrationResult)
async def run_agents(request: Request, body: OrchestrationRequest):
    engine = request.app.state.engine
    try:
        return await orchestrate(engine, body)
    except NoAvailableModelError as e:
        raise HTTPException(status_code=503, detail=_error_body(str(e), code=503)) from e
    except PlannerError as e:
        raise HTTPException(status_code=422, detail=_error_body(str(e), code=422)) from e
    except OrchestrationError as e:
        raise HTTPException(status_code=503, detail=_error_body(str(e), code=503)) from e
    except DagValidationError as e:
        # Should be unreachable -- generate_plan() already validates the
        # exact same graph before returning (see app/api/plan.py's
        # identical defensive comment). Kept as a defensive net, never
        # trusted to be impossible.
        raise HTTPException(
            status_code=500, detail=_error_body(f"orchestrator produced a plan that failed DAG validation unexpectedly: {e}", code=500)
        ) from e
