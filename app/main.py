"""FastAPI application entrypoint. `uvicorn app.main:app` or `scripts/start.sh`."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import admin, dag, health, models, openai_compatible, plan
from app.core.config import get_settings
from app.core.engine import ChatEngine
from app.core.lifecycle import shutdown, startup
from app.observability.logging import get_logger, setup_logging

setup_logging()
logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    ctx = await startup(settings)
    app.state.context = ctx
    app.state.engine = ChatEngine(ctx)
    yield
    await shutdown(ctx)


app = FastAPI(title="XRouter", description="OpenAI-compatible AI intelligence gateway", version="0.1.0", lifespan=lifespan)

_settings_for_cors = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings_for_cors.server.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request size limit + bounded global timeout (section 三十三) ----------

@app.middleware("http")
async def request_guard(request: Request, call_next):
    settings = get_settings()
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > settings.server.max_request_body_bytes:
        return JSONResponse(status_code=413, content={"error": {"message": "Request body too large", "type": "xrouter_error"}})
    return await call_next(request)


# --- Simple per-client (IP) rate limiter, in-memory sliding window --------

_rate_buckets: dict[str, deque] = defaultdict(deque)


@app.middleware("http")
async def rate_limit_guard(request: Request, call_next):
    if not request.url.path.startswith("/v1/"):
        return await call_next(request)
    settings = get_settings()
    client = request.client.host if request.client else "unknown"
    now = time.time()
    bucket = _rate_buckets[client]
    bucket.append(now)
    while bucket and (now - bucket[0]) > 60:
        bucket.popleft()
    if len(bucket) > settings.server.per_client_rate_limit_per_minute:
        return JSONResponse(status_code=429, content={"error": {"message": "Rate limit exceeded", "type": "xrouter_error"}})
    return await call_next(request)


app.include_router(health.router)
app.include_router(models.router)
app.include_router(openai_compatible.router)
app.include_router(admin.router)
app.include_router(dag.router)
app.include_router(plan.router)


@app.get("/")
async def root():
    return {"name": "XRouter", "docs": "/docs", "base_url": "/v1"}
