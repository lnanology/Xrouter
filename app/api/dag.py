"""POST /v1/dag/run — execute a client-supplied DAG of chat-completion
nodes. See app/execution/dag.py and app/contracts/dag.py for the model."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.contracts.dag import DagRunRequest
from app.execution.dag import DagExecutor, DagValidationError
from app.observability.logging import get_logger

router = APIRouter()
logger = get_logger("api.dag")


def _error_body(message: str, type_: str = "xrouter_error", code: int = 400) -> dict:
    return {"message": message, "type": type_, "code": code}


@router.post("/v1/dag/run")
async def run_dag(request: Request, body: DagRunRequest):
    ctx = request.app.state.context
    engine = request.app.state.engine
    executor = DagExecutor(
        engine, max_nodes=ctx.settings.routing.max_dag_nodes,
        tools=ctx.tools, max_tool_iterations=ctx.settings.routing.max_tool_iterations,
        max_critique_retries=ctx.settings.routing.max_critique_retries,
    )
    try:
        return await executor.run(body)
    except DagValidationError as e:
        raise HTTPException(status_code=400, detail=_error_body(str(e))) from e
