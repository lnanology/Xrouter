"""Phase 2 Task Classifier (section 三十六). Builds on the Phase 1 complexity
heuristic (app.intelligence.complexity) by also guessing a coarse task type,
then uses (task_type, complexity) to *suggest* a routing policy.

Important scope boundary: this still never triggers multi-agent
orchestration or a different execution pipeline — Phase 1's single
fast-path engine (app.core.engine.ChatEngine) is unchanged. All this adds
is a smarter default for *which policy* the adaptive router uses when the
client didn't explicitly pick one, instead of always falling back to the
static `routing.default_policy` from config. A client-supplied
`routing_policy` always wins over this suggestion.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.contracts.request import ChatCompletionRequest
from app.intelligence.complexity import classify_complexity

_CODE_KEYWORDS = (
    "```", "def ", "class ", "function", "compile", "stack trace", "traceback",
    "regex", "sql", "api", "code", "程式", "bug", "unit test", "refactor",
)
_RESEARCH_KEYWORDS = (
    "research", "sources", "compare", "latest", "news", "find out", "cite",
    "according to", "統計", "資料",
)
_REASONING_KEYWORDS = (
    "prove", "why", "logic", "step by step", "solve", "calculate", "derive",
    "reason through",
)
_CREATIVE_KEYWORDS = (
    "write a story", "poem", "creative", "imagine", "fiction", "brainstorm",
    "slogan", "故事",
)


class TaskType(str, Enum):
    CHAT = "chat"
    CODE = "code"
    RESEARCH = "research"
    REASONING = "reasoning"
    CREATIVE = "creative"
    TOOL_USE = "tool_use"


@dataclass
class TaskClassification:
    task_type: TaskType
    complexity: int  # 0..4, from app.intelligence.complexity
    requires_tools: bool
    requires_vision: bool
    suggested_policy: str


def _detect_task_type(text: str, has_tools: bool) -> TaskType:
    if has_tools:
        return TaskType.TOOL_USE
    lowered = text.lower()
    # Order matters: code/research/reasoning cues are checked before the
    # broader creative bucket so e.g. "write a function" doesn't get
    # misread as creative writing just because it starts with "write".
    if any(kw in lowered for kw in _CODE_KEYWORDS):
        return TaskType.CODE
    if any(kw in lowered for kw in _RESEARCH_KEYWORDS):
        return TaskType.RESEARCH
    if any(kw in lowered for kw in _REASONING_KEYWORDS):
        return TaskType.REASONING
    if any(kw in lowered for kw in _CREATIVE_KEYWORDS):
        return TaskType.CREATIVE
    return TaskType.CHAT


def _suggest_policy(task_type: TaskType, complexity: int) -> str:
    if complexity >= 3:
        return "quality"
    if task_type == TaskType.TOOL_USE:
        return "reliable"
    if task_type == TaskType.CODE and complexity >= 2:
        return "quality"
    if task_type == TaskType.RESEARCH:
        return "reliable"
    if complexity <= 1 and task_type == TaskType.CHAT:
        return "fastest"
    return "balanced"


def classify(request: ChatCompletionRequest) -> TaskClassification:
    text = " ".join((m.content if isinstance(m.content, str) else "") for m in request.messages)
    has_tools = bool(request.tools)
    complexity = classify_complexity(request)
    task_type = _detect_task_type(text, has_tools)
    requires_vision = any(
        isinstance(m.content, list) and any(part.get("type") == "image_url" for part in m.content if isinstance(part, dict))
        for m in request.messages
    )

    return TaskClassification(
        task_type=task_type,
        complexity=complexity,
        requires_tools=has_tools,
        requires_vision=requires_vision,
        suggested_policy=_suggest_policy(task_type, complexity),
    )
