"""OpenAI-compatible response contracts."""
from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import BaseModel, Field


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: dict[str, Any]
    finish_reason: str | None = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage = Field(default_factory=Usage)
    system_fingerprint: str | None = None
    xrouter: dict[str, Any] | None = None  # provider/model/policy metadata


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: dict[str, Any]
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChunkChoice]


def extract_message_text(response: ChatCompletionResponse) -> str:
    """Pulls the plain-text content out of a response's first choice, or
    "" if there isn't any (e.g. a tool-call-only response). Shared by
    anything that needs to *read* a completed response's answer rather
    than just relay it — the quality gate (app/intelligence/quality_gate.py)
    and the DAG executor's {{node_id}} output substitution
    (app/execution/dag.py) both use this instead of each re-implementing
    the same "choices[0].message.get('content')" reach-in."""
    if not response.choices:
        return ""
    content = response.choices[0].message.get("content")
    return content if isinstance(content, str) else ""
