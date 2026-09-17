"""Confidence Engine contracts (Phase 4, last piece, spec 三十六). See
app/intelligence/confidence.py for how an assessment is actually produced --
a pure, local, deterministic roll-up of signals a completed Orchestrator run
already computed, distinct from app/intelligence/quality_gate.py's own
per-call structural check (see that module's docstring for the division)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ConfidenceAssessment(BaseModel):
    score: float  # 0.0-1.0
    label: str  # "high" | "medium" | "low"
    reasons: list[str] = Field(default_factory=list)
