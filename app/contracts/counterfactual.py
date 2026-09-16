"""Counterfactual contracts (Phase 4, spec 三十六). See
app/intelligence/counterfactual.py for how an analysis is actually built,
and its own module docstring for why this is scoped to pure reasoning
about an answer's own assumptions rather than actually re-running
anything (that's Simulation's job, the next Phase 4 piece)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class CounterfactualPoint(BaseModel):
    # A concrete, load-bearing premise the final answer actually depends
    # on -- not a trivial or unfalsifiable one.
    assumption: str
    # How the answer would meaningfully change if this assumption turned
    # out to be false.
    if_false: str


class CounterfactualAnalysis(BaseModel):
    points: list[CounterfactualPoint] = Field(default_factory=list)
