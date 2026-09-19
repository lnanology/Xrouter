"""Quality Gate (Phase 2, spec 三十六): a cheap, local,
deterministic pass over a completed response that catches the failure
modes XRouter can actually observe without paying for another model call —
an empty reply, output the provider itself truncated, or a small local
model stuck in a degenerate repetition loop. This structural check
(assess(), below) is deliberately NOT a judgment of factual correctness —
that needs a second model call to grade the first.

That second call is this module's other half: LLM-graded judgment
(build_llm_judge_request()/parse_llm_judgment(), below), gated by its own
routing.quality_gate_llm_grading_enabled switch (off by default, nested
inside the already-opt-in routing.quality_gate_enabled — a real extra
provider call per assessment must never turn on silently, same "must not
turn on silently" rule as every other cost-bearing switch in this
codebase). When on, ChatEngine._apply_quality_gate (app/core/engine.py)
runs it only after the structural check already passed — a structurally
broken response doesn't need a second opinion — and treats a "not
satisfied" LLM verdict the same as a structural failure, retrying the
next untried candidate. This mirrors the forced-tool-call pattern Critic
(app/intelligence/critic.py) and Verifier (app/intelligence/verifier.py)
already established (a "submit_<x>" forced tool call, fail-open on any
error), with one deliberate difference: the judge call here bypasses
ChatEngine.handle_chat entirely (calling the router + run_chat directly
instead, exactly like _apply_quality_gate's own candidate-retry logic
already does) because, unlike Critic/Verifier, this call happens *from
inside* _apply_quality_gate itself — routing it back through handle_chat
would re-enter _apply_quality_gate for the judge's own response and,
if LLM grading is on, try to grade the judge's grading, unboundedly.

If routing.quality_gate_enabled is on and an assessment fails (structural
or, when enabled, LLM-graded), the engine retries with the next untried
candidate from the same routing decision, up to routing.max_quality_retries
times, and always returns the best answer it found rather than erroring
out — a low-quality answer beats no answer, consistent with XRouter's
graceful-degradation principle. Every assessment (pass or fail) is
attached to the response's xrouter.quality metadata so callers can see
what happened.

Non-streaming only: a streamed response has already been sent to the
client chunk by chunk by the time it could be assessed, so there is
nothing left to retry (same reasoning as the first-chunk-only fallback
boundary in app/routing/fallback.py's run_stream_chat).

Distinct from the later, differently-scoped Confidence Engine
(app/intelligence/confidence.py, Phase 4's last piece): that module never
re-reads response text itself, it rolls up an already-finished
Orchestrator run's own higher-level signals (Critic/Verifier/Debate/
Evidence Graph outcomes) into one confidence score, with no extra
provider call and no retry of its own."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from app.contracts.request import ChatCompletionRequest, ChatMessage
from app.contracts.response import ChatCompletionResponse, extract_message_text

MIN_CONTENT_CHARS = 2
REPETITION_WORD_MIN_COUNT = 6              # don't judge very short replies for word repetition
REPETITION_WORD_RATIO_THRESHOLD = 0.5      # one word making up >=50% of all words is degenerate
REPETITION_CHAR_MIN_LEN = 20               # don't judge very short replies for char repetition
REPETITION_CHAR_UNIQUE_RATIO_THRESHOLD = 0.15  # <15% unique characters in a long reply


@dataclass
class QualityAssessment:
    passed: bool
    score: float
    reasons: list[str] = field(default_factory=list)
    # Set only when LLM grading ran and returned feedback (i.e. it found
    # the response unsatisfactory) -- None for a purely structural
    # assessment, or when the LLM judge was satisfied.
    llm_feedback: str | None = None


def _has_degenerate_word_repetition(text: str) -> bool:
    """Catches loops like 'the the the the the...' or a model stuck
    repeating one token/word far more than natural language ever would."""
    words = text.split()
    if len(words) < REPETITION_WORD_MIN_COUNT:
        return False
    _, top_count = Counter(words).most_common(1)[0]
    return (top_count / len(words)) >= REPETITION_WORD_RATIO_THRESHOLD


def _has_degenerate_char_repetition(text: str) -> bool:
    """Catches loops at the character level ('aaaaaaaaaa...',
    '!!!!!!!!!!...') that word-splitting wouldn't necessarily catch."""
    stripped = text.strip()
    if len(stripped) < REPETITION_CHAR_MIN_LEN:
        return False
    unique_ratio = len(set(stripped)) / len(stripped)
    return unique_ratio <= REPETITION_CHAR_UNIQUE_RATIO_THRESHOLD


def _tool_call_was_forced(request: ChatCompletionRequest) -> bool:
    tc = request.tool_choice
    if isinstance(tc, dict):
        return True
    return tc == "required"


def assess(response: ChatCompletionResponse, request: ChatCompletionRequest, min_score: float = 0.5) -> QualityAssessment:
    reasons: list[str] = []
    score = 1.0
    text = extract_message_text(response)
    message = response.choices[0].message if response.choices else {}
    finish_reason = response.choices[0].finish_reason if response.choices else None
    has_tool_calls = bool(message.get("tool_calls"))

    # A tool call *is* the model's answer — empty text content alongside one
    # is normal and must not be penalized as an empty response.
    if not has_tool_calls and len(text.strip()) < MIN_CONTENT_CHARS:
        reasons.append("empty_response")
        score -= 0.7

    if finish_reason == "length":
        reasons.append("truncated_output")
        score -= 0.3

    if text and _has_degenerate_word_repetition(text):
        reasons.append("degenerate_word_repetition")
        score -= 0.6

    if text and _has_degenerate_char_repetition(text):
        reasons.append("degenerate_char_repetition")
        score -= 0.6

    if _tool_call_was_forced(request) and not has_tool_calls:
        reasons.append("missing_forced_tool_call")
        score -= 0.6

    score = max(0.0, min(1.0, score))
    return QualityAssessment(passed=score >= min_score, score=round(score, 3), reasons=reasons)


# --- LLM-graded judgment (opt-in, see module docstring) ---------------------

LLM_QUALITY_TOOL_NAME = "submit_quality_judgment"

_LLM_QUALITY_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": LLM_QUALITY_TOOL_NAME,
        "description": "Report whether a response is a good, correct, complete answer to the request it was given.",
        "parameters": {
            "type": "object",
            "properties": {
                "satisfied": {
                    "type": "boolean",
                    "description": "true if the response is accurate, complete, and actually answers the request; "
                                    "false if it has factual errors, is incomplete, off-topic, or unhelpful.",
                },
                "feedback": {
                    "type": "string",
                    "description": "If not satisfied, a specific, actionable description of what's wrong. Empty string if satisfied.",
                },
            },
            "required": ["satisfied", "feedback"],
        },
    },
}

_MAX_JUDGED_CHARS = 4000  # mirrors critic.py's own _MAX_OUTPUT_CHARS


@dataclass
class LLMQualityJudgment:
    satisfied: bool
    feedback: str | None = None


def _format_original_request(request: ChatCompletionRequest) -> str:
    """Renders the request's messages as compact text for the judge
    prompt. Non-string (multimodal) content is skipped, the same
    convention app/execution/critique_loop.py, app/intelligence/
    task_classifier.py, and app/cache/manager.py already use for this
    exact kind of text-only analysis."""
    lines = [f"{m.role}: {m.content}" for m in request.messages if isinstance(m.content, str)]
    return "\n".join(lines)


def build_llm_judge_request(
    request: ChatCompletionRequest, response: ChatCompletionResponse, routing_policy: str | None,
) -> ChatCompletionRequest | None:
    """None when there's nothing worth LLM-grading -- a tool-call response
    with no text content, the same exemption assess() already gives that
    case (a tool call *is* the model's answer)."""
    text = extract_message_text(response).strip()
    message = response.choices[0].message if response.choices else {}
    if not text and message.get("tool_calls"):
        return None

    system = (
        "You are the quality-judgment stage of XRouter, an AI gateway. You "
        "will be shown the original request and the response a model "
        "produced for it. Judge only whether that response is accurate, "
        f"complete, and actually answers the request -- call {LLM_QUALITY_TOOL_NAME} "
        "with your judgment. Do not answer the request yourself."
    )
    user = f"Original request:\n{_format_original_request(request)}\n\nResponse to judge:\n{text[:_MAX_JUDGED_CHARS]}"
    messages = [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)]
    return ChatCompletionRequest(
        model="auto", messages=messages, routing_policy=routing_policy,
        tools=[_LLM_QUALITY_TOOL_SCHEMA], tool_choice={"type": "function", "function": {"name": LLM_QUALITY_TOOL_NAME}},
    )


def _extract_tool_arguments(response: ChatCompletionResponse) -> str | None:
    """Independently duplicated from critic.py/verifier.py's own identical
    helper, matching their own precedent of not sharing it."""
    if not response.choices:
        return None
    message = response.choices[0].message
    tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
    if not tool_calls:
        return None
    fn = tool_calls[0].get("function", {}) if isinstance(tool_calls[0], dict) else {}
    return fn.get("arguments")


def parse_llm_judgment(judge_response: ChatCompletionResponse) -> LLMQualityJudgment:
    """Fail-open, mirroring critic.py/verifier.py's own parsing exactly: a
    missing or malformed submit_quality_judgment call is treated as
    satisfied -- a best-effort second opinion should never block or
    endlessly retry a response that already passed the structural check."""
    raw = _extract_tool_arguments(judge_response)
    if raw is None:
        return LLMQualityJudgment(satisfied=True, feedback=None)
    try:
        data = json.loads(raw)
        satisfied = bool(data["satisfied"])
    except (json.JSONDecodeError, TypeError, KeyError):
        return LLMQualityJudgment(satisfied=True, feedback=None)
    return LLMQualityJudgment(satisfied=satisfied, feedback=data.get("feedback") or None)
