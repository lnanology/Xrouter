"""Critic retry loop (Phase 3): when a DAG node opts into node.critique,
after the node produces a response this asks the Critic
(app/intelligence/critic.py) whether it satisfies the node's own
instruction; if not, it re-runs the node's own request with the critic's
feedback appended, bounded by max_retries, before accepting whatever the
last attempt produced -- the same "never block an already-produced result
forever" fail-open discipline the Verifier follows for a whole run,
scoped here to one node.

`responder` is the node's own way of actually getting an answer -- plain
ChatEngine.handle_chat, or app/execution/tool_loop.py's run_with_tools
when the node also has enable_tools -- passed in rather than decided here,
so this loop never needs to know or care whether tools are involved; it
only ever re-runs through the exact same path the node already used."""
from __future__ import annotations

from typing import TYPE_CHECKING, Awaitable, Callable

from app.contracts.critic import CritiqueResult
from app.contracts.request import ChatCompletionRequest, ChatMessage
from app.contracts.response import ChatCompletionResponse
from app.intelligence.critic import critique as run_critique
from app.observability.logging import get_logger

if TYPE_CHECKING:
    from app.core.engine import ChatEngine

logger = get_logger("execution.critique_loop")

DEFAULT_MAX_CRITIQUE_RETRIES = 1

Responder = Callable[[ChatCompletionRequest], Awaitable[ChatCompletionResponse]]


def _node_prompt_text(request: ChatCompletionRequest) -> str:
    return "\n".join(m.content for m in request.messages if isinstance(m.content, str))


async def run_with_critique(
    engine: "ChatEngine", request: ChatCompletionRequest, response: ChatCompletionResponse,
    responder: Responder, routing_policy: str | None = None, max_retries: int = DEFAULT_MAX_CRITIQUE_RETRIES,
) -> tuple[ChatCompletionResponse, CritiqueResult]:
    prompt_text = _node_prompt_text(request)
    result = await run_critique(engine, prompt_text, response, routing_policy)

    current = request
    attempts = 0
    while not result.satisfied and attempts < max_retries:
        attempts += 1
        logger.info("critique attempt %d/%d unsatisfied: %s", attempts, max_retries, result.feedback)
        current = current.model_copy(update={"messages": [
            *current.messages,
            ChatMessage(
                role="user",
                content=f"A reviewer found an issue with your last answer: {result.feedback or 'it did not satisfy the instruction'}. Provide a corrected answer.",
            ),
        ]})
        response = await responder(current)
        result = await run_critique(engine, prompt_text, response, routing_policy)

    return response, result
