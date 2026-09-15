"""POST /v1/research -- a standalone research question (Phase 3, spec
section 十九's Researcher). Thin HTTP wrapper around
app/agents/researcher.py; see that module for how the answer is actually
produced and why an unconfigured web_search tool degrades gracefully
instead of failing the request."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.agents.researcher import research
from app.contracts.researcher import ResearchRequest, ResearchResult
from app.core.errors import NoAvailableModelError

router = APIRouter()


def _error_body(message: str, type_: str = "xrouter_error", code: int = 400) -> dict:
    return {"message": message, "type": type_, "code": code}


@router.post("/v1/research", response_model=ResearchResult)
async def run_research(request: Request, body: ResearchRequest):
    ctx = request.app.state.context
    engine = request.app.state.engine
    try:
        return await research(engine, body.query, ctx.tools, routing_policy=body.routing_policy)
    except NoAvailableModelError as e:
        raise HTTPException(status_code=503, detail=_error_body(str(e), code=503)) from e
