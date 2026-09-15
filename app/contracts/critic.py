"""Critic contracts (Phase 3): per-node review, distinct from the
Verifier (app/contracts/verifier.py), which only ever judges a *whole* DAG
run against the *original* task, once, after everything has finished. See
app/intelligence/critic.py for how a critique is actually produced, and
app/execution/critique_loop.py for how a node retries itself with the
critic's feedback folded in."""
from __future__ import annotations

from pydantic import BaseModel


class CritiqueResult(BaseModel):
    satisfied: bool
    feedback: str | None = None
