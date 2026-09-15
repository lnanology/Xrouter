"""DAG (directed acyclic graph) execution contracts (Phase 2, spec 三十六).

A DAG run is a client-supplied set of chat-completion nodes with explicit
dependencies. XRouter does not invent a Planner that decides the graph for
you — auto-generating a DAG from a single free-form request is Phase 3+
multi-agent orchestration (Planner/Researcher/Critic/Verifier), which does
not exist yet. What exists here is the execution substrate that phase
will eventually sit on top of: hand XRouter an explicit graph, and it runs
independent nodes concurrently, respects dependencies, and feeds each
node's own upstream outputs into it via `{{node_id}}` placeholders in that
node's message content."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.contracts.request import ChatMessage
from app.contracts.response import ChatCompletionResponse


class DagNodeRequest(BaseModel):
    id: str
    depends_on: list[str] = Field(default_factory=list)
    messages: list[ChatMessage]
    model: str = "auto"
    routing_policy: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None


class DagRunRequest(BaseModel):
    nodes: list[DagNodeRequest]


NodeStatus = Literal["success", "failed", "skipped"]


class DagNodeResult(BaseModel):
    id: str
    status: NodeStatus
    response: ChatCompletionResponse | None = None
    error: str | None = None
    latency_ms: float | None = None


class DagRunResponse(BaseModel):
    id: str
    status: Literal["success", "partial", "failed"]
    nodes: list[DagNodeResult]
    latency_ms: float
