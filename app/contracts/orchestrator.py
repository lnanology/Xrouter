"""Orchestrator contracts (Phase 3: Dynamic Agent Team, spec section
十九). See app/agents/orchestrator.py for how a team is actually
assembled and run, and app/api/orchestrator.py for POST /v1/agents/run."""
from __future__ import annotations

from pydantic import BaseModel

from app.contracts.dag import DagRunResponse
from app.contracts.verifier import VerificationResult


class OrchestrationRequest(BaseModel):
    task: str
    context: str | None = None
    model: str = "auto"
    routing_policy: str | None = None
    # Groups this run's memory reads/writes (app/storage/repositories/
    # memory.py) -- "global" by default. Set to a session/user id to keep
    # recall scoped to that conversation rather than leaking across
    # unrelated callers.
    scope: str = "global"


class OrchestrationResult(BaseModel):
    id: str
    # Which stages actually ran, e.g. ["solver"] or
    # ["planner", "specialists", "critic", "research", "verifier", "synthesizer"]
    # -- always reflects reality, never what a tier "should" have used.
    team: list[str]
    complexity: int  # 0..4, from the Task Classifier
    task_type: str
    answer: str  # the final, user-facing answer -- always a plain string, whatever team ran
    dag: DagRunResponse | None = None  # set only when the team ran more than a bare single call (tier >= 2)
    verification: VerificationResult | None = None  # set only at tier 4 (mandatory verification)
    latency_ms: float
