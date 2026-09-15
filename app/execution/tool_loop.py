"""Tool-execution loop (Phase 3): when a DAG/Planner node opts into
XRouter-executed tools (DagNodeRequest.enable_tools), this drives the
call -> tool_call -> execute -> feed result back -> call again cycle,
bounded by max_iterations, entirely on top of the existing
ChatEngine.handle_chat() -- no separate provider-calling logic, the same
reuse principle every other execution module follows (race, dag,
planner, verifier).

Only a tool call naming a *registered, configured* tool is executed
automatically; any other tool call in the response (a client-supplied
tool the node's own `tools`/`tool_choice` also listed) is left untouched
and returned as-is -- XRouter never silently "answers" a tool call it
has no registered executor for. Ends gracefully rather than looping
forever: after max_iterations, whatever the model last said is returned,
tool call or not."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from app.contracts.request import ChatCompletionRequest, ChatMessage
from app.contracts.response import ChatCompletionResponse
from app.core.errors import ToolError
from app.observability.logging import get_logger
from app.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from app.core.engine import ChatEngine

logger = get_logger("execution.tool_loop")

DEFAULT_MAX_TOOL_ITERATIONS = 3


async def run_with_tools(
    engine: "ChatEngine", request: ChatCompletionRequest, tools: ToolRegistry,
    enabled_names: list[str], max_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
) -> ChatCompletionResponse:
    executable = {name: tools.get(name) for name in enabled_names if tools.get(name) is not None}
    missing = [name for name in enabled_names if name not in executable]
    if missing:
        logger.info("requested tool(s) %s are not registered/configured; ignoring", missing)
    if not executable:
        # Nothing requested is actually registered/configured -- behave
        # exactly like a plain node, rather than silently advertising
        # tools that don't exist.
        return await engine.handle_chat(request)

    schemas = [tool.schema for tool in executable.values()]
    current = request.model_copy(update={
        "tools": [*(request.tools or []), *schemas],
        "tool_choice": request.tool_choice or "auto",
    })

    response = await engine.handle_chat(current)
    iterations = 0
    while iterations < max_iterations:
        message = response.choices[0].message if response.choices else {}
        tool_calls = message.get("tool_calls") or []
        runnable = [tc for tc in tool_calls if tc.get("function", {}).get("name") in executable]
        if not runnable:
            return response

        tool_messages: list[ChatMessage] = []
        for tc in runnable:
            name = tc["function"]["name"]
            tool = executable[name]
            try:
                arguments = json.loads(tc["function"].get("arguments") or "{}")
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            try:
                result_text = await tool.execute(arguments)
            except ToolError as e:
                # Feed the failure back to the model as the tool's own
                # result rather than failing the node -- the model can
                # often recover (rephrase the query, answer without it).
                logger.warning("tool '%s' failed: %s", name, e)
                result_text = f"Tool error: {e}"
            tool_messages.append(ChatMessage(role="tool", tool_call_id=tc.get("id"), content=result_text))

        assistant_message = ChatMessage(role="assistant", content=message.get("content"), tool_calls=tool_calls)
        current = current.model_copy(update={"messages": [*current.messages, assistant_message, *tool_messages]})
        response = await engine.handle_chat(current)
        iterations += 1

    return response
