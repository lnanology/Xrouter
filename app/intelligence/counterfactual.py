"""Counterfactual (Phase 4, spec 三十六 -- the third of the five Phase 4
pieces; see README's Phase 4 section for the rest).

Given a finished answer, identifies the load-bearing assumptions it
actually depends on -- premises that, if different, would meaningfully
change the answer, not trivial or unfalsifiable ones -- and for each,
describes concretely how the answer would change if that assumption
turned out to be false. This is pure analysis of an answer that already
exists: it never re-runs anything, never perturbs an input and re-
executes the DAG. That's deliberately left to Simulation (Phase 4's next,
still-unbuilt piece) -- building actual re-execution machinery here,
before Simulation's own scope is even designed, would be exactly the
"不要新增不必要的infrastructure" (spec rule 15) this project keeps
refusing to do.

Unlike Evidence Graph (Phase 4's first piece), which traces claims to
specific DAG nodes and is therefore a structural no-op below tier 2 (no
DAG exists yet at tier 0-1), Counterfactual's question -- "what does this
answer's own reasoning depend on" -- only needs the task and the final
answer. It genuinely works at every complexity tier, including 0-1. When
a DAG did run, its own summarize_dag() rendering is folded in as extra
grounding (the fifth module to reuse it, after the Verifier, Synthesizer,
Evidence Graph and Debate), but it's optional context, never a
requirement.

Counterfactual is not named in spec section 十九's own "Very hard" team
table (unlike Debate, which is), so -- same reasoning as Evidence Graph --
it ships opt-in via OrchestrationRequest.trace_counterfactual rather than
running automatically at any tier.

Same "no fake placeholder functionality" principle as the rest of this
package: an actual LLM judgment (a forced "submit_counterfactual_analysis"
tool call), never a heuristic. Same fail-open philosophy as Evidence
Graph too: a missing/malformed tool call, or NoAvailableModelError from
the call itself, returns an empty CounterfactualAnalysis rather than
blocking or retrying an already-produced answer -- this is advisory
metadata about an answer XRouter already committed to returning, never
something that should hold it hostage or revise it (that's Debate's job,
a different piece with a different, answer-revising contract)."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from app.contracts.counterfactual import CounterfactualAnalysis, CounterfactualPoint
from app.contracts.dag import DagRunResponse
from app.contracts.request import ChatCompletionRequest, ChatMessage
from app.contracts.response import ChatCompletionResponse
from app.core.errors import NoAvailableModelError
from app.execution.dag_summary import summarize_dag
from app.observability.logging import get_logger

if TYPE_CHECKING:
    from app.core.engine import ChatEngine

logger = get_logger("intelligence.counterfactual")

COUNTERFACTUAL_TOOL_NAME = "submit_counterfactual_analysis"

_COUNTERFACTUAL_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": COUNTERFACTUAL_TOOL_NAME,
        "description": "Report the final answer's own load-bearing assumptions, and how the answer would change if each one didn't hold.",
        "parameters": {
            "type": "object",
            "properties": {
                "points": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "assumption": {
                                "type": "string",
                                "description": "One concrete, load-bearing assumption/premise the answer actually depends on -- not a trivial or unfalsifiable one.",
                            },
                            "if_false": {
                                "type": "string",
                                "description": "How the answer would meaningfully change if this assumption turned out to be false.",
                            },
                        },
                        "required": ["assumption", "if_false"],
                    },
                },
            },
            "required": ["points"],
        },
    },
}


def _build_counterfactual_request(
    task: str, answer: str, context: str | None, dag: DagRunResponse | None, routing_policy: str | None,
) -> ChatCompletionRequest:
    system = (
        "You are the counterfactual-analysis stage of XRouter, an AI "
        "gateway. You will be shown the user's original task and the "
        "final answer that was given. Identify the answer's own "
        "load-bearing assumptions -- premises that, if different, would "
        "actually change the answer, not trivial or unfalsifiable ones -- "
        "and for each, describe concretely how the answer would change if "
        "it didn't hold. If the answer genuinely has no meaningful "
        f"assumptions to challenge, call {COUNTERFACTUAL_TOOL_NAME} with "
        "an empty points list rather than inventing one. Call "
        f"{COUNTERFACTUAL_TOOL_NAME} with your findings -- do not answer "
        "the task yourself."
    )
    parts = [f"Original task:\n{task}", f"Final answer:\n{answer}"]
    if context:
        parts.append(f"Context:\n{context}")
    if dag is not None:
        parts.append(f"What each step produced:\n{summarize_dag(dag)}")
    user = "\n\n".join(parts)
    messages = [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)]
    return ChatCompletionRequest(
        model="auto", messages=messages, routing_policy=routing_policy,
        tools=[_COUNTERFACTUAL_TOOL_SCHEMA],
        tool_choice={"type": "function", "function": {"name": COUNTERFACTUAL_TOOL_NAME}},
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


def _sanitize_points(raw_points: list[dict]) -> list[CounterfactualPoint]:
    points: list[CounterfactualPoint] = []
    for raw in raw_points:
        try:
            assumption = raw["assumption"]
            if_false = raw["if_false"]
        except (KeyError, TypeError):
            continue
        points.append(CounterfactualPoint(assumption=assumption, if_false=if_false))
    return points


async def build_counterfactual_analysis(
    engine: "ChatEngine",
    task: str,
    answer: str,
    context: str | None = None,
    dag: DagRunResponse | None = None,
    routing_policy: str | None = None,
) -> CounterfactualAnalysis:
    request = _build_counterfactual_request(task, answer, context, dag, routing_policy)
    try:
        response = await engine.handle_chat(request)
    except NoAvailableModelError as e:
        logger.warning("counterfactual analysis could not get an answer (%s); returning an empty analysis", e)
        return CounterfactualAnalysis(points=[])

    raw = _extract_tool_arguments(response)
    if raw is None:
        logger.warning("counterfactual analysis did not call %s; returning an empty analysis", COUNTERFACTUAL_TOOL_NAME)
        return CounterfactualAnalysis(points=[])

    try:
        data = json.loads(raw)
        raw_points = data["points"]
        if not isinstance(raw_points, list):
            raise TypeError("points was not a list")
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        logger.warning("counterfactual analysis returned an unusable %s call (%s); returning an empty analysis", COUNTERFACTUAL_TOOL_NAME, e)
        return CounterfactualAnalysis(points=[])

    return CounterfactualAnalysis(points=_sanitize_points(raw_points))
