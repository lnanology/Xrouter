"""Planner contracts (Phase 3 groundwork, multi-agent orchestration): a
single free-form task in, an explicit app.contracts.dag.DagRunRequest-
shaped graph out. See app/intelligence/planner.py for how the graph is
actually generated, and app/api/plan.py for POST /v1/plan/run, which
generates the plan and then runs it through the exact same DagExecutor a
client-supplied DAG uses."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.contracts.dag import DagRunResponse
from app.contracts.verifier import VerificationResult


class PlanRequest(BaseModel):
    task: str
    context: str | None = None
    model: str = "auto"                 # which model does the *planning*, not which model(s) execute the resulting nodes
    routing_policy: str | None = None   # overrides the server's configured planner_routing_policy for this one call
    max_nodes: int | None = None        # falls back to routing.max_dag_nodes
    # Verifier (Phase 3 groundwork): off by default, same reason as race
    # mode/quality gate -- a failed verification costs an entire extra
    # plan+execute cycle to retry, not just one call, so it shouldn't turn
    # on silently. When true, the DAG's result is checked against the
    # original task by app/intelligence/verifier.py; if not satisfied, the
    # Planner re-plans with the verifier's feedback folded in and the
    # whole thing re-runs, up to routing.max_verify_retries times.
    verify: bool = False


class PlanNodeSpec(BaseModel):
    id: str
    depends_on: list[str] = Field(default_factory=list)
    prompt: str

    @field_validator("depends_on", mode="before")
    @classmethod
    def _coerce_none_to_empty(cls, v):
        # Real models occasionally emit "depends_on": null for "no
        # dependencies" instead of omitting the key or sending [] --
        # tolerate that rather than failing the whole plan over it.
        return v or []


class PlanSpec(BaseModel):
    nodes: list[PlanNodeSpec]


class PlanRunResponse(BaseModel):
    id: str
    plan: PlanSpec
    plan_attempts: int          # 1 == the model got a valid, runnable plan on the first try
    dag: DagRunResponse
    verification: VerificationResult | None = None   # set only when the request opted into verify=true
    replan_count: int = 0        # how many times the Verifier's feedback triggered a fresh plan+execute cycle
    latency_ms: float
