"""Confidence / Quality-Gate Engine (Phase 2, spec 三十六): a cheap, local,
deterministic pass over a completed response that catches the failure
modes XRouter can actually observe without paying for another model call —
an empty reply, output the provider itself truncated, or a small local
model stuck in a degenerate repetition loop. This is deliberately NOT a
judgment of factual correctness (that needs a second model call to grade
the first, which XRouter does not spend money on by default) — only
structural red flags a request/response pair can reveal on their own.

If routing.quality_gate_enabled is on and an assessment fails, the engine
(app/core/engine.py, ChatEngine._apply_quality_gate) retries with the next
untried candidate from the same routing decision, up to
routing.max_quality_retries times, and always returns the best answer it
found rather than erroring out — a low-quality answer beats no answer,
consistent with XRouter's graceful-degradation principle. Every assessment
(pass or fail) is attached to the response's xrouter.quality metadata so
callers can see what happened.

Non-streaming only: a streamed response has already been sent to the
client chunk by chunk by the time it could be assessed, so there is
nothing left to retry (same reasoning as the first-chunk-only fallback
boundary in app/routing/fallback.py's run_stream_chat)."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from app.contracts.request import ChatCompletionRequest
from app.contracts.response import ChatCompletionResponse

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


def _extract_text(response: ChatCompletionResponse) -> str:
    if not response.choices:
        return ""
    content = response.choices[0].message.get("content")
    return content if isinstance(content, str) else ""


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
    text = _extract_text(response)
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
