"""POST /v1/chat/completions — the main OpenAI-compatible endpoint. Clients
never need to know which provider actually served the request; that's
entirely the router's decision (section 四十)."""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.contracts.request import ChatCompletionRequest
from app.core.engine import ChatEngine
from app.core.errors import NoAvailableModelError
from app.observability.logging import get_logger

router = APIRouter()
logger = get_logger("api.chat")


def _error_body(message: str, type_: str = "xrouter_error", code: int = 503) -> dict:
    return {"error": {"message": message, "type": type_, "code": code}}


@router.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest):
    ctx = request.app.state.context
    engine: ChatEngine = request.app.state.engine

    if body.stream:
        async def event_stream():
            try:
                async for chunk in engine.handle_stream_chat(body):
                    yield f"data: {chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
            except NoAvailableModelError as e:
                logger.warning("stream failed: %s", e)
                err = _error_body(str(e))
                yield f"data: {json.dumps(err)}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    try:
        response = await engine.handle_chat(body)
    except NoAvailableModelError as e:
        raise HTTPException(status_code=503, detail=_error_body(str(e))["error"]) from e

    return response
