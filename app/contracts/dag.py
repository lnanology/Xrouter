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
for its own one planning call. A node can also opt into two further,
per-node behaviors on top of the plain chat-completion call: enable_tools
(XRouter-executed tools, app/execution/tool_loop.py) and critique
(per-node review, app/intelligence/critic.py + app/execution/
critique_loop.py) -- distinct from the Verifier (app/intelligence/
verifier.py), which only ever judges the *whole* run against the
*original* task, once, at the very end."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.contracts.critic import CritiqueResult
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
    # Names of XRouter-registered tools (app/tools/registry.py, e.g.
    # "web_search") this node may use. Distinct from `tools` above, which
    # is passed straight through for the *client* to execute, per the
    # standard OpenAI contract -- a name listed here is instead executed
    # by XRouter itself (app/execution/tool_loop.py): the call -> tool ->
    # call round-trip happens inside this one node, bounded by
    # routing.max_tool_iterations, before the node's result comes back.
    enable_tools: list[str] = Field(default_factory=list)
    # Per-node review (Phase 3, app/intelligence/critic.py,
    # app/execution/critique_loop.py): off by default, same reasoning as
    # race mode/the quality gate/verify -- an unsatisfied critique costs
    # an extra call *and* re-runs this node, so it shouldn't turn on
    # silently. When true, right after this node produces a response, the
    # Critic judges it against this node's own instruction (not the wider
    # task); if unsatisfied, the node re-runs with the critic's feedback
    # folded in, up to routing.max_critique_retries times, before
    # accepting whatever the last attempt produced.
    critique: bool = False


class DagRunRequest(BaseModel):
    nodes: list[DagNodeRequest]


NodeStatus = Literal["success", "failed", "skipped"]


class DagNodeResult(BaseModel):
    id: str
    status: NodeStatus
    response: ChatCompletionResponse | None = None
    error: str | None = None
    latency_ms: float | None = None
    # Set only when this node had critique=true -- the Critic's final
    # judgment (after any retries), for visibility into whether/how many
    # times this specific node's output was reviewed and redone.
    critique: CritiqueResult | None = None


class DagRunResponse(BaseModel):
    id: str
    status: Literal["success", "partial", "failed"]
    nodes: list[DagNodeResult]
    latency_ms: float
