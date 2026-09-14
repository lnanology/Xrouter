"""GET /v1/models — OpenAI-compatible model listing."""
from __future__ import annotations

import time

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/v1/models")
async def list_models(request: Request):
    ctx = request.app.state.context
    data = [
        {
            "id": m.public_id(),
            "object": "model",
            "created": int(time.time()),
            "owned_by": m.provider_id,
        }
        for m in ctx.models.all()
        if m.active
    ]
    return {"object": "list", "data": data}
