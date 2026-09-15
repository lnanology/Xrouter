"""Critic (Phase 3, last of the three orchestration pieces alongside the
richer Planner node spec and the Web Search tool): per-node review. The
Verifier (app/intelligence/verifier.py) judges the *whole run* against the
*original* task, exactly once, after everything has finished -- useful,
but a bad step three nodes deep can already have poisoned everything
downstream (via {{node_id}} substitution) long before the Verifier ever
gets a look. The Critic instead reviews one node's own output against
that node's own instruction, right after the node produces it, so a weak
step can be caught and redone on the spot -- see
app/execution/critique_loop.py for the retry loop this feeds.

Same "no fake placeholder functionality" principle as the Planner and
Verifier: an actual LLM judgment (a forced "submit_critique" tool call),
never a fixed heuristic or keyword check. Same fail-open philosophy as
the Verifier, too: a missing/malformed tool call, or NoAvailableModelError
from the critique call itself, counts as "satisfied" rather than blocking
or endlessly retrying a node that already produced *something* -- a
best-effort review should never hold a node's result hostage."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from app.contracts.critic import CritiqueResult
from app.contracts.request import ChatCompletionRequest, ChatMessage
from app.contracts.response import ChatCompletionResponse, extract_message_text
from app.core.errors import NoAvailableModelError
from app.observability.logging import get_logger

if TYPE_CHECKING:
    from app.core.engine import ChatEngine

logger = get_logger("intelligence.critic")

CRITIQUE_TOOL_NAME = "submit_critique"

_CRITIQUE_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": CRITIQUE_TOOL_NAME,
        "description": "Report whether a single step's own output actually satisfies that step's own instruction.",
        "parameters": {
            "type": "object",
            "properties": {
                "satisfied": {
                    "type": "boolean",
                    "description": "true if the output genuinely satisfies the step's own instruction, false if something is missing or wrong.",
                },
                "feedback": {
                    "type": "string",
                    "description": "If not satisfied, a specific, actionable description of what's missing or wrong, so the step can be redone. Empty string if satisfied.",
                },
            },
            "required": ["satisfied", "feedback"],
        },
    },
}

_MAX_OUTPUT_CHARS = 4000  # keep one node's output from crowding out the rest of the critic's own context


def _build_critique_request(prompt: str, output: str, routing_policy: str | None) -> ChatCompletionRequest:
    system = (
        "You are the critic stage of XRouter, an AI gateway. You will be "
        "shown one step's own instruction and the output it actually "
        "produced. Judge only whether that output satisfies *this* "
        "instruction -- not the wider task it's part of, just this one "
        "step. Call submit_critique with your judgment -- do not answer "
        "the instruction yourself."
    )
    user = f"Step instruction:\n{prompt}\n\nStep output:\n{output[:_MAX_OUTPUT_CHARS]}"
    messages = [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)]
    return ChatCompletionRequest(
        model="auto", messages=messages, routing_policy=routing_policy,
        tools=[_CRITIQUE_TOOL_SCHEMA], tool_choice={"type": "function", "function": {"name": CRITIQUE_TOOL_NAME}},
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


async def critique(
    engine: "ChatEngine", prompt: str, response: ChatCompletionResponse, routing_policy: str | None = None,
) -> CritiqueResult:
    output = extract_message_text(response).strip()
    request = _build_critique_request(prompt, output, routing_policy)
    try:
        critique_response = await engine.handle_chat(request)
    except NoAvailableModelError as e:
        logger.warning("critic could not get an answer (%s); treating the step as satisfied (fail-open)", e)
        return CritiqueResult(satisfied=True, feedback=None)

    raw = _extract_tool_arguments(critique_response)
    if raw is None:
        logger.warning("critic did not call submit_critique; treating the step as satisfied (fail-open)")
        return CritiqueResult(satisfied=True, feedback=None)

    try:
        data = json.loads(raw)
        satisfied = bool(data["satisfied"])
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        logger.warning("critic returned an unusable submit_critique call (%s); treating the step as satisfied (fail-open)", e)
        return CritiqueResult(satisfied=True, feedback=None)

    feedback = data.get("feedback") or None
    return CritiqueResult(satisfied=satisfied, feedback=feedback)
