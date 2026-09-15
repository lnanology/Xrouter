"""DAG Executor (Phase 2, spec 三十六): runs a client-supplied graph of
chat-completion nodes, respecting explicit dependencies. Each node is
executed through the exact same ChatEngine.handle_chat() every other
XRouter request goes through, so DAG nodes get full routing, caching,
circuit-breaking, race mode, and the quality gate for free — this module
only adds scheduling (topological waves) and `{{node_id}}` output
substitution on top; it never talks to a provider directly.

Execution is wave-based (Kahn's-algorithm layering): all nodes whose
dependencies are already resolved run concurrently in one wave via
asyncio.gather, then the next wave is computed from what just finished.
This is the simplest *correct* schedule for a DAG — a node with a fast,
already-satisfied dependency can still wait out a slower sibling in the
same wave before its own dependents start, where a fully event-driven
ready-queue scheduler could shave that idle time off. Wave-based is what
ships first (see README 'not implemented yet' for the refinement); it is
never wrong, only occasionally not maximally fast.

If a node fails, everything that (transitively) depends on it is marked
"skipped" rather than attempted — a downstream node whose prompt is
supposed to reference a failed upstream node's output has nothing
meaningful to substitute in, so running it anyway would be worse than not
running it. Nodes unrelated to the failure still run normally.

A node can opt into two further, per-node behaviors layered on top of the
plain handle_chat() call, in `_run_node`'s `responder`: `enable_tools`
(app/execution/tool_loop.py — XRouter itself executes a bounded call ->
tool -> call loop) and `critique` (app/execution/critique_loop.py — the
Critic, app/intelligence/critic.py, reviews the node's own output against
its own instruction and the node re-runs through the same `responder` if
unsatisfied). Neither one duplicates handle_chat()'s own routing/caching/
circuit-breaking logic; both just wrap the same call."""
from __future__ import annotations

import asyncio
import re
import time
from typing import TYPE_CHECKING

from app.contracts.dag import DagNodeRequest, DagNodeResult, DagRunRequest, DagRunResponse
from app.contracts.request import ChatCompletionRequest
from app.contracts.response import extract_message_text
from app.core.errors import NoAvailableModelError, XRouterError
from app.execution.critique_loop import DEFAULT_MAX_CRITIQUE_RETRIES, run_with_critique
from app.execution.tool_loop import DEFAULT_MAX_TOOL_ITERATIONS, run_with_tools
from app.tools.registry import ToolRegistry
from app.utils.ids import new_id

if TYPE_CHECKING:
    from app.core.engine import ChatEngine

_PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z0-9_\-]+)\s*\}\}")


class DagValidationError(XRouterError):
    """The graph itself is malformed (cycle, unknown dependency, duplicate
    id, too many nodes) — raised before anything is executed, so a bad DAG
    never burns a single provider call."""


def _validate_and_order(nodes: list[DagNodeRequest], max_nodes: int) -> list[list[DagNodeRequest]]:
    if not nodes:
        raise DagValidationError("A DAG run needs at least one node.")
    if len(nodes) > max_nodes:
        raise DagValidationError(f"DAG has {len(nodes)} nodes, exceeding the configured limit of {max_nodes}.")

    by_id: dict[str, DagNodeRequest] = {}
    for n in nodes:
        if n.id in by_id:
            raise DagValidationError(f"Duplicate node id: '{n.id}'.")
        by_id[n.id] = n

    for n in nodes:
        for dep in n.depends_on:
            if dep == n.id:
                raise DagValidationError(f"Node '{n.id}' cannot depend on itself.")
            if dep not in by_id:
                raise DagValidationError(f"Node '{n.id}' depends on unknown node '{dep}'.")

    in_degree = {n.id: len(n.depends_on) for n in nodes}
    dependents: dict[str, list[str]] = {n.id: [] for n in nodes}
    for n in nodes:
        for dep in n.depends_on:
            dependents[dep].append(n.id)

    waves: list[list[DagNodeRequest]] = []
    remaining = dict(in_degree)
    while remaining:
        ready_ids = [nid for nid, deg in remaining.items() if deg == 0]
        if not ready_ids:
            raise DagValidationError(f"DAG has a cycle involving: {', '.join(sorted(remaining))}.")
        waves.append([by_id[nid] for nid in ready_ids])
        for nid in ready_ids:
            del remaining[nid]
            for dependent in dependents[nid]:
                remaining[dependent] -= 1

    return waves


def _substitute(text: str, outputs: dict[str, str]) -> str:
    def _replace(match: re.Match) -> str:
        return outputs.get(match.group(1), match.group(0))

    return _PLACEHOLDER_RE.sub(_replace, text)


def _build_request(node: DagNodeRequest, outputs: dict[str, str]) -> ChatCompletionRequest:
    messages = [
        m.model_copy(update={"content": _substitute(m.content, outputs)}) if isinstance(m.content, str) else m
        for m in node.messages
    ]
    return ChatCompletionRequest(
        model=node.model, messages=messages, routing_policy=node.routing_policy,
        temperature=node.temperature, max_tokens=node.max_tokens, tools=node.tools, tool_choice=node.tool_choice,
    )


class DagExecutor:
    def __init__(
        self, engine: "ChatEngine", max_nodes: int = 20,
        tools: ToolRegistry | None = None, max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
        max_critique_retries: int = DEFAULT_MAX_CRITIQUE_RETRIES,
    ):
        self._engine = engine
        self._max_nodes = max_nodes
        # An empty registry (the default) makes any node's enable_tools
        # simply resolve to nothing available -- graceful degradation,
        # not a crash, matching every other "asked for something
        # unconfigured" path in XRouter.
        self._tools = tools if tools is not None else ToolRegistry()
        self._max_tool_iterations = max_tool_iterations
        self._max_critique_retries = max_critique_retries

    async def run(self, dag: DagRunRequest) -> DagRunResponse:
        start = time.time()
        waves = _validate_and_order(dag.nodes, self._max_nodes)

        results: dict[str, DagNodeResult] = {}
        outputs: dict[str, str] = {}
        unavailable: set[str] = set()  # failed or skipped — anything depending on these gets skipped too

        for wave in waves:
            runnable = [n for n in wave if not any(dep in unavailable for dep in n.depends_on)]
            runnable_ids = {n.id for n in runnable}
            for n in wave:
                if n.id not in runnable_ids:
                    results[n.id] = DagNodeResult(id=n.id, status="skipped", error="an upstream dependency failed or was skipped")
                    unavailable.add(n.id)

            if not runnable:
                continue

            wave_results = await asyncio.gather(*(self._run_node(n, outputs) for n in runnable))
            for n, result in zip(runnable, wave_results, strict=True):
                results[n.id] = result
                if result.status == "failed":
                    unavailable.add(n.id)
                else:
                    outputs[n.id] = extract_message_text(result.response) if result.response else ""

        ordered = [results[n.id] for n in dag.nodes]
        statuses = {r.status for r in ordered}
        overall = "success" if statuses == {"success"} else ("failed" if "success" not in statuses else "partial")

        return DagRunResponse(id=new_id("dag"), status=overall, nodes=ordered, latency_ms=round((time.time() - start) * 1000, 1))

    async def _run_node(self, node: DagNodeRequest, outputs: dict[str, str]) -> DagNodeResult:
        node_start = time.time()
        request = _build_request(node, outputs)

        async def responder(req: ChatCompletionRequest):
            if node.enable_tools:
                return await run_with_tools(
                    self._engine, req, self._tools, node.enable_tools, max_iterations=self._max_tool_iterations,
                )
            return await self._engine.handle_chat(req)

        try:
            response = await responder(request)
            critique_result = None
            if node.critique:
                # Reruns through the exact same responder (plain call, or
                # the tool loop when enable_tools is also set) -- the
                # critique loop never needs its own provider-calling
                # logic, same reuse principle every execution module here
                # follows.
                response, critique_result = await run_with_critique(
                    self._engine, request, response, responder,
                    routing_policy=node.routing_policy, max_retries=self._max_critique_retries,
                )
            return DagNodeResult(
                id=node.id, status="success", response=response, critique=critique_result,
                latency_ms=round((time.time() - node_start) * 1000, 1),
            )
        except NoAvailableModelError as e:
            return DagNodeResult(id=node.id, status="failed", error=str(e), latency_ms=round((time.time() - node_start) * 1000, 1))
