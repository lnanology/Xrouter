"""Verifier contracts (Phase 3 groundwork). See app/intelligence/verifier.py
for how a verification is actually produced, and
app/execution/plan_runner.py for how it feeds a possible re-plan."""
from __future__ import annotations

from pydantic import BaseModel


class VerificationResult(BaseModel):
    satisfied: bool
    feedback: str | None = None
