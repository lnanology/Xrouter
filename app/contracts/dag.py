"""DAG (directed acyclic graph) execution contracts (Phase 2, spec 三十六).

A DAG run is a set of chat-completion nodes with explicit dependencies:
hand XRouter a graph, and it runs independent nodes concurrently, respects
dependencies, and feeds each node's own upstream outputs into it via
`{{node_id}}` placeholders in that node's message content. This module
defines the execution substrate only -- deciding *what* the graph should
be is a separate concern: a client can supply one directly (POST
/v1/dag/run, app/api/dag.py), or the Planner (Phase 3 groundwork,
app/intelligence/planner.py) can generate one from a single free-form
task and hand it to this exact same executor (POST /v1/plan/run,
app/api/plan.py). Neither path duplicates the other's logic; the Planner
only ever produces a DagRunRequest, it never talks to a provider except
for its own one planning call."""
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
