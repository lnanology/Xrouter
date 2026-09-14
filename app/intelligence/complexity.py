"""Task-complexity heuristic (section 七). This is the canonical
implementation; `app.routing.classifier` re-exports it for backward
compatibility with Phase 1 call sites.

Deliberately a heuristic, not a model call: it must be near-instant since it
runs on every request before routing. In Phase 1 it only ever informs a
speed/quality bias in policy selection — it never triggers multi-agent
orchestration (that's Phase 3's agents/ module, not yet built)."""
from __future__ import annotations

from app.contracts.request import ChatCompletionRequest

HARD_KEYWORDS = (
    "step by step", "architecture", "design a", "prove", "algorithm", "optimi",
    "research", "compare and contrast", "debug", "refactor", "multi-step",
)


def classify_complexity(request: ChatCompletionRequest) -> int:
    """Returns 0 (trivial) .. 4 (very hard)."""
    text = " ".join((m.content if isinstance(m.content, str) else "") for m in request.messages)
    length = len(text)
    n_messages = len(request.messages)
    has_tools = bool(request.tools)
    hard_hits = sum(1 for kw in HARD_KEYWORDS if kw in text.lower())

    score = 0
    if length > 200:
        score += 1
    if length > 800:
        score += 1
    if n_messages > 6:
        score += 1
    if has_tools:
        score += 1
    if hard_hits >= 1:
        score += 1
    if hard_hits >= 3:
        score += 1

    return min(score, 4)
