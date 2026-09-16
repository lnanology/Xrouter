"""Orchestrator contracts (Phase 3: Dynamic Agent Team, spec section
十九). See app/agents/orchestrator.py for how a team is actually
assembled and run, and app/api/orchestrator.py for POST /v1/agents/run."""
from __future__ import annotations

from pydantic import BaseModel, Field

from app.contracts.counterfactual import CounterfactualAnalysis
from app.contracts.dag import DagRunResponse
from app.contracts.debate import DebateResult
from app.contracts.evidence import EvidenceGraph
from app.contracts.simulation import SimulationRun
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
    # Evidence Graph (Phase 4, app/intelligence/evidence.py): opt-in, same
    # reasoning as verify/critique/race -- an extra LLM call shouldn't
    # turn on silently. A silent no-op at tier 0-1, since there's no DAG
    # there to trace any claim against.
    trace_evidence: bool = False
    # Counterfactual (Phase 4, app/intelligence/counterfactual.py):
    # opt-in, same reasoning as trace_evidence -- Counterfactual isn't a
    # spec-named standing team member (unlike Debate), so it never turns
    # on silently. Unlike trace_evidence, it genuinely runs at every
    # tier, including 0-1, since it needs only the task and the final
    # answer, not a DAG.
    trace_counterfactual: bool = False
    # Simulation (Phase 4, app/agents/simulation.py): opt-in caller-
    # supplied list of changed-premise scenarios to actually re-answer
    # the task under, not another spec-named standing team member either.
    # Empty by default -- a no-op unless the caller supplies scenarios.
    simulate: list[str] = Field(default_factory=list)


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
    debate: DebateResult | None = None  # set only at tier 4, and only when the debate actually completed (fails open otherwise)
    evidence: EvidenceGraph | None = None  # set only when trace_evidence was requested and a DAG actually ran
    counterfactual: CounterfactualAnalysis | None = None  # set only when trace_counterfactual was requested -- unlike evidence, can be set at any tier
    simulations: list[SimulationRun] = Field(default_factory=list)  # one entry per scenario in `simulate` that actually produced an answer
    latency_ms: float
