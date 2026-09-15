"""Researcher (Phase 3, agents/ per spec section 十九): a focused,
reusable research step. Not a new execution mechanism -- it reuses
app/execution/tool_loop.py's run_with_tools() exactly like a DAG node's
own enable_tools does -- just a distinct, purpose-built system prompt and
persona for genuinely investigating a question, which a bare DAG node
(whatever prompt a client or the Planner happened to write) doesn't get.
Reachable directly (POST /v1/research, app/api/research.py) for a
standalone research question, and used by app/agents/orchestrator.py's
higher-complexity tiers.

Graceful degradation, not a fake answer: if web_search isn't registered
and configured, this still answers from the model's own knowledge --
honestly, since the system prompt tells the model plainly whether a
search tool is available -- rather than fabricating results or refusing
outright. See ResearchResult.tool_available for what actually happened."""
from __future__ import annotations

from typing import TYPE_CHECKING

from app.contracts.request import ChatCompletionRequest, ChatMessage
from app.contracts.researcher import ResearchResult
from app.contracts.response import extract_message_text
from app.execution.tool_loop import DEFAULT_MAX_TOOL_ITERATIONS, run_with_tools
from app.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from app.core.engine import ChatEngine

_SEARCH_TOOL_NAME = "web_search"


def _build_research_request(question: str, has_search: bool, routing_policy: str | None) -> ChatCompletionRequest:
    system = (
        "You are the research stage of XRouter, an AI gateway. Investigate "
        "the following question thoroughly and answer it directly, citing "
        "sources (titles/URLs) when you use them. "
        + (
            "You have a web_search tool -- use it whenever the question needs "
            "current or external information you can't already be sure of."
            if has_search else
            "No web search tool is available right now -- answer from your own "
            "knowledge, and say plainly if you're not confident or the "
            "information could be out of date."
        )
    )
    return ChatCompletionRequest(
        model="auto", routing_policy=routing_policy,
        messages=[ChatMessage(role="system", content=system), ChatMessage(role="user", content=question)],
    )


async def research(
    engine: "ChatEngine", question: str, tools: ToolRegistry, routing_policy: str | None = None,
    max_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
) -> ResearchResult:
    has_search = tools.get(_SEARCH_TOOL_NAME) is not None
    request = _build_research_request(question, has_search, routing_policy)

    if has_search:
        response = await run_with_tools(engine, request, tools, [_SEARCH_TOOL_NAME], max_iterations=max_iterations)
    else:
        response = await engine.handle_chat(request)

    return ResearchResult(answer=extract_message_text(response).strip(), tool_available=has_search)
