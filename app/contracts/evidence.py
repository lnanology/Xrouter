"""Evidence Graph contracts (Phase 4, spec 三十六). See
app/intelligence/evidence.py for how a graph is actually built, and its
own module docstring for why "graph" here means a flat claim -> DAG-node
mapping, not literal multi-hop graph traversal."""
from __future__ import annotations

from pydantic import BaseModel, Field


class EvidenceClaim(BaseModel):
    claim: str
    # DAG node ids that back this claim, or ["model_knowledge"] when the
    # claim isn't traceable to any step's own output. Never trusted
    # blindly -- app/intelligence/evidence.py drops any id here that
    # isn't actually a real node id (or "model_knowledge") rather than
    # propagating a hallucinated reference.
    supported_by: list[str] = Field(default_factory=list)
    # The model's own judgment: does what supported_by cites actually
    # back this claim, or is it asserted without real support even though
    # a step ran? Distinct from supported_by being empty (nothing cited
    # at all) -- a claim can cite a node and still be judged unsupported
    # if that node's own output doesn't actually say it.
    supported: bool = True


class EvidenceGraph(BaseModel):
    claims: list[EvidenceClaim] = Field(default_factory=list)
