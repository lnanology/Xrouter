"""OpenAI-compatible request contracts (pydantic, used directly by FastAPI)."""
from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: list[str] | str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    user: str | None = None

    # XRouter extensions (all optional, ignored by plain OpenAI clients)
    routing_policy: str | None = None
    x_cache: bool | None = None


class XRouterRequestContext(BaseModel):
    """Internal envelope threaded through the routing/execution pipeline."""

    request_id: str = Field(default_factory=lambda: f"req_{uuid.uuid4().hex[:20]}")
    created_at: float = Field(default_factory=time.time)
    complexity: int = 0
    routing_policy: str = "balanced"
    client_id: str | None = None
