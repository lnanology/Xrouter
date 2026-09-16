"""Simulation contracts (Phase 4, spec 三十六). See
app/agents/simulation.py for how a scenario is actually run, and its own
module docstring for why this is scoped to actually re-executing a
caller-supplied changed premise rather than guessing how the answer
would change (that's Counterfactual's job, the previous Phase 4 piece)."""
from __future__ import annotations

from pydantic import BaseModel


class SimulationRun(BaseModel):
    # The caller-supplied changed premise, echoed back so a response
    # with multiple runs is unambiguous about which is which.
    scenario: str
    # What the task's answer actually becomes under that premise --
    # a real re-execution, not a description of how it might change.
    answer: str
