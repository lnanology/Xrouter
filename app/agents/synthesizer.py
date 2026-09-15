"""Synthesizer (Phase 3, agents/ per spec section 十九): composes ONE
direct, coherent final answer from a completed DAG run's own node
outputs, so a caller of the Orchestrator (app/agents/orchestrator.py)
never has to reassemble a multi-node result themselves. Distinct from the
Verifier (judges whether the run satisfied the task) and the Critic
(reviews one node's own output) -- this is the one stage whose whole job
is producing the actual user-facing answer.

Not a forced-tool-call structured-output stage like the Planner/Verifier/
Critic -- it just writes prose, so there is no honest "best effort"
fallback this module can invent on its own if the underlying call fails
(NoAvailableModelError propagates untouched). app/agents/orchestrator.py
catches that and falls back to the DAG's own best terminal output instead
of losing an already-produced result -- the fallback lives with the
caller, who actually has something to fall back to, not here."""
from __future__ import annotations

from typing import TYPE_CHECKING

from app.contracts.dag import DagRunResponse
from app.contracts.request import ChatCompletionRequest, ChatMessage
from app.contracts.response import ChatCompletionResponse
from app.execution.dag_summary import summarize_dag

if TYPE_CHECKING:
    from app.core.engine import ChatEngine


def _build_synthesis_request(
    task: str, context: str | None, dag: DagRunResponse, routing_policy: str | None,
) -> ChatCompletionRequest:
    system = (
        "You are the synthesis stage of XRouter, an AI gateway. You will be "
        "shown the user's original task and the results of the steps taken "
        "to accomplish it. Write ONE clear, direct, well-organized final "
        "answer for the user -- do not mention the steps, the plan, or that "
        "this was broken into multiple parts unless the user's task itself "
        "asked for that. Just answer, as if you had done it all yourself."
    )
    parts = [f"Original task:\n{task}"]
    if context:
        parts.append(f"Context:\n{context}")
    parts.append(f"Step results:\n{summarize_dag(dag)}")
    messages = [ChatMessage(role="system", content=system), ChatMessage(role="user", content="\n\n".join(parts))]
    return ChatCompletionRequest(model="auto", routing_policy=routing_policy, messages=messages)


async def synthesize(
    engine: "ChatEngine", task: str, context: str | None, dag: DagRunResponse, routing_policy: str | None = None,
) -> ChatCompletionResponse:
    request = _build_synthesis_request(task, context, dag, routing_policy)
    return await engine.handle_chat(request)
