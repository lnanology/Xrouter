"""Debate contracts (Phase 4, spec 三十六 -- the second piece, per spec
section 十九's own "Very hard" team table: Planner + parallel specialists
+ Research + Debate + Verification + Synthesizer). See
app/agents/debate.py for how a debate is actually run."""
from __future__ import annotations

from pydantic import BaseModel


class DebateResult(BaseModel):
    position: str    # the draft answer that went into the debate
    advocate: str     # the strongest good-faith case FOR the draft
    skeptic: str        # the strongest good-faith case AGAINST the draft
    resolution: str      # the judge's final, strengthened answer -- becomes OrchestrationResult.answer when debate ran
