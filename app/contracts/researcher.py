"""Researcher contracts (Phase 3, agents/ per spec section 十九). See
app/agents/researcher.py for how a research answer is actually produced."""
from __future__ import annotations

from pydantic import BaseModel


class ResearchRequest(BaseModel):
    query: str
    routing_policy: str | None = None


class ResearchResult(BaseModel):
    answer: str
    # Whether a real search tool (e.g. web_search) was actually registered
    # and configured for this call -- NOT whether the model chose to
    # invoke it (tool use is the model's own judgment, same as any
    # enable_tools node; XRouter never forces a tool call it can't
    # justify). False means the answer came from the model's own
    # knowledge alone.
    tool_available: bool
