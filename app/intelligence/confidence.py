"""Confidence Engine (Phase 4, last piece, spec 三十六): a pure, local,
deterministic roll-up of everything a completed Orchestrator run already
observed about its own final answer -- which layers of scrutiny actually ran
(per-node Critic, whole-run Verifier, Debate, Evidence Graph), and whether
each one that ran was satisfied -- into one confidence signal for the
caller. This is deliberately NOT a second LLM judgment: it never re-reads
response text itself, it only combines already-computed structured results.

Distinct from app/intelligence/quality_gate.py's own "Quality Gate", which
is a single-call, structural-defect check (empty/truncated/repetitive text,
or a forced tool call that didn't happen) wired into the fast routing path
to trigger a same-candidate retry. This module never looks at raw response
text and never causes an extra provider call -- it only rolls up signals a
finished Orchestrator run already produced, for every run, automatically
(unlike Evidence Graph/Counterfactual/Simulation, which are opt-in because
each spends a real extra provider call; this one spends nothing extra, so
gating it behind a flag would be arbitrary).

Notably not async, and takes no ChatEngine -- the one Phase 3/4 intelligence
module with neither, underscoring that it makes zero provider calls."""
from __future__ import annotations

from app.contracts.confidence import ConfidenceAssessment
from app.contracts.dag import DagRunResponse
from app.contracts.debate import DebateResult
from app.contracts.evidence import EvidenceGraph
from app.contracts.verifier import VerificationResult

# dag is None (tier 0-1): nothing reviewed the answer at all, so this is an
# honest "no signal" midpoint, not a guess dressed up as a real score.
UNREVIEWED_SCORE = 0.5

_CRITIQUE_UNSATISFIED_PENALTY = 0.3
_DAG_PARTIAL_PENALTY = 0.3
_VERIFICATION_UNSATISFIED_PENALTY = 0.3
_DEBATE_FAILED_OPEN_PENALTY = 0.1
_UNSUPPORTED_CLAIMS_MAX_PENALTY = 0.3  # scaled by unsupported/total ratio


def assess_confidence(
    team: list[str],
    dag: DagRunResponse | None = None,
    verification: VerificationResult | None = None,
    debate: DebateResult | None = None,
    evidence: EvidenceGraph | None = None,
) -> ConfidenceAssessment:
    if dag is None:
        return ConfidenceAssessment(score=UNREVIEWED_SCORE, label="medium", reasons=["unreviewed"])

    reasons: list[str] = []
    score = 1.0

    if dag.status != "success":
        reasons.append("dag_partial_or_failed")
        score -= _DAG_PARTIAL_PENALTY

    if any(n.critique is not None and not n.critique.satisfied for n in dag.nodes):
        reasons.append("critique_unsatisfied")
        score -= _CRITIQUE_UNSATISFIED_PENALTY

    if verification is not None and not verification.satisfied:
        reasons.append("verification_unsatisfied")
        score -= _VERIFICATION_UNSATISFIED_PENALTY

    if "verifier" in team and debate is None:
        reasons.append("debate_failed_open")
        score -= _DEBATE_FAILED_OPEN_PENALTY

    if evidence is not None and evidence.claims:
        unsupported = sum(1 for c in evidence.claims if not c.supported)
        if unsupported:
            reasons.append("unsupported_claims")
            score -= _UNSUPPORTED_CLAIMS_MAX_PENALTY * (unsupported / len(evidence.claims))

    score = max(0.0, min(1.0, round(score, 3)))
    label = "high" if score >= 0.8 else "medium" if score >= 0.5 else "low"
    return ConfidenceAssessment(score=score, label=label, reasons=reasons)
