"""Verifier (Phase 3 groundwork): the piece that closes the loop the
Planner and DAG executor started. Given the original task and what a
generated DAG actually produced, asks an LLM (a forced "submit_verification"
tool call, the same structured-output pattern the Planner uses) whether
the run genuinely accomplished the task -- an actual model judgment, not a
fixed heuristic, consistent with XRouter's "no fake placeholder
functionality" principle. See app/execution/plan_runner.py for how a
failed verification triggers a bounded re-plan-and-re-execute loop.

Fails open: if the verifier's own tool call is missing or malformed, or
if no provider can even answer it, that is treated as "satisfied" (with
a note in the logs) rather than blocking or looping forever -- a
plan+execute cycle that already finished should not be held hostage by a
best-effort second opinion. This mirrors the quality gate's own "never
error out, always return the best answer found" philosophy (see
app/intelligence/quality_gate.py)."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from app.contracts.dag import DagRunResponse
from app.contracts.request import ChatCompletionRequest, ChatMessage
from app.contracts.response import ChatCompletionResponse, extract_message_text
from app.contracts.verifier import VerificationResult
from app.core.errors import NoAvailableModelError
from app.observability.logging import get_logger

if TYPE_CHECKING:
    from app.core.engine import ChatEngine

logger = get_logger("intelligence.verifier")

VERIFY_TOOL_NAME = "submit_verification"

_VERIFY_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": VERIFY_TOOL_NAME,
        "description": "Report whether the completed run actually accomplished the user's original task.",
        "parameters": {
            "type": "object",
            "properties": {
                "satisfied": {
                    "type": "boolean",
                    "description": "true if the task was accomplished, false if something is missing or wrong.",
                },
                "feedback": {
                    "type": "string",
                    "description": "If not satisfied, a specific, actionable description of what's missing or wrong, so a new plan can fix it. Empty string if satisfied.",
                },
            },
            "required": ["satisfied", "feedback"],
        },
    },
}

_MAX_NODE_OUTPUT_CHARS = 2000  # keep a single node's output from crowding out everything else in the verifier's context


def _summarize_dag(dag: DagRunResponse) -> str:
    lines: list[str] = []
    for node in dag.nodes:
        if node.status == "success" and node.response is not None:
            text = extract_message_text(node.response).strip()
            lines.append(f"- '{node.id}' (success): {text[:_MAX_NODE_OUTPUT_CHARS]}")
        elif node.status == "failed":
            lines.append(f"- '{node.id}' (failed): {node.error}")
        else:
            lines.append(f"- '{node.id}' (skipped): {node.error}")
    return "\n".join(lines) or "(no nodes ran)"


def _build_verification_request(
    task: str, context: str | None, dag: DagRunResponse, routing_policy: str | None,
) -> ChatCompletionRequest:
    system = (
        "You are the verification stage of XRouter, an AI gateway. You "
        "will be shown the user's original task and what a generated plan "
        "actually produced, step by step. Decide whether the task was "
        "genuinely accomplished. Call submit_verification with your "
        "judgment -- do not answer the task yourself."
    )
    parts = [f"Original task:\n{task}"]
    if context:
        parts.append(f"Context:\n{context}")
    parts.append(f"What the plan produced:\n{_summarize_dag(dag)}")
    messages = [ChatMessage(role="system", content=system), ChatMessage(role="user", content="\n\n".join(parts))]
    return ChatCompletionRequest(
        model="auto", messages=messages, routing_policy=routing_policy,
        tools=[_VERIFY_TOOL_SCHEMA], tool_choice={"type": "function", "function": {"name": VERIFY_TOOL_NAME}},
    )


def _extract_tool_arguments(response: ChatCompletionResponse) -> str | None:
    if not response.choices:
        return None
    message = response.choices[0].message
    tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
    if not tool_calls:
        return None
    fn = tool_calls[0].get("function", {}) if isinstance(tool_calls[0], dict) else {}
    return fn.get("arguments")


async def verify(
    engine: "ChatEngine", task: str, context: str | None, dag: DagRunResponse, routing_policy: str | None = None,
) -> VerificationResult:
    request = _build_verification_request(task, context, dag, routing_policy)
    try:
        response = await engine.handle_chat(request)
    except NoAvailableModelError as e:
        logger.warning("verifier could not get an answer (%s); treating the run as satisfied (fail-open)", e)
        return VerificationResult(satisfied=True, feedback=None)

    raw = _extract_tool_arguments(response)
    if raw is None:
        logger.warning("verifier did not call submit_verification; treating the run as satisfied (fail-open)")
        return VerificationResult(satisfied=True, feedback=None)

    try:
        data = json.loads(raw)
        satisfied = bool(data["satisfied"])
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        logger.warning("verifier returned an unusable submit_verification call (%s); treating the run as satisfied (fail-open)", e)
        return VerificationResult(satisfied=True, feedback=None)

    feedback = data.get("feedback") or None
    return VerificationResult(satisfied=satisfied, feedback=feedback)
