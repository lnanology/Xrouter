from app.contracts.critic import CritiqueResult
from app.contracts.dag import DagNodeResult, DagRunResponse
from app.contracts.debate import DebateResult
from app.contracts.evidence import EvidenceClaim, EvidenceGraph
from app.contracts.verifier import VerificationResult
from app.intelligence.confidence import UNREVIEWED_SCORE, assess_confidence


def _dag(status: str = "success", nodes: list[DagNodeResult] | None = None) -> DagRunResponse:
    return DagRunResponse(id="run_1", status=status, nodes=nodes or [], latency_ms=1.0)


def _node(id: str = "solve", status: str = "success", critique: CritiqueResult | None = None) -> DagNodeResult:
    return DagNodeResult(id=id, status=status, critique=critique)


def _debate() -> DebateResult:
    return DebateResult(position="draft", advocate="for", skeptic="against", resolution="final")


# --- tier 0-1: no dag at all -------------------------------------------------

def test_no_dag_returns_the_unreviewed_baseline():
    result = assess_confidence(["solver"])
    assert result.score == UNREVIEWED_SCORE == 0.5
    assert result.label == "medium"
    assert result.reasons == ["unreviewed"]


# --- clean run ----------------------------------------------------------------

def test_clean_dag_run_scores_perfect_with_no_reasons():
    dag = _dag(status="success", nodes=[_node(critique=CritiqueResult(satisfied=True))])
    result = assess_confidence(["solver", "critic"], dag=dag)
    assert result.score == 1.0
    assert result.label == "high"
    assert result.reasons == []


# --- each penalty in isolation ------------------------------------------------

def test_dag_partial_status_is_penalized():
    dag = _dag(status="partial", nodes=[_node()])
    result = assess_confidence(["solver"], dag=dag)
    assert "dag_partial_or_failed" in result.reasons
    assert result.score == 0.7


def test_critique_unsatisfied_is_penalized():
    dag = _dag(nodes=[_node(critique=CritiqueResult(satisfied=False, feedback="not quite"))])
    result = assess_confidence(["solver", "critic"], dag=dag)
    assert "critique_unsatisfied" in result.reasons
    assert result.score == 0.7


def test_verification_unsatisfied_is_penalized():
    dag = _dag(nodes=[_node()])
    verification = VerificationResult(satisfied=False, feedback="missed something")
    # debate supplied so the debate_failed_open penalty doesn't also fire here --
    # this test isolates the verification penalty alone.
    result = assess_confidence(
        ["planner", "specialists", "critic", "verifier", "debate"],
        dag=dag, verification=verification, debate=_debate(),
    )
    assert "verification_unsatisfied" in result.reasons
    assert result.score == 0.7


def test_debate_failed_open_is_penalized_when_verifier_ran_without_a_debate_result():
    dag = _dag(nodes=[_node()])
    result = assess_confidence(["planner", "specialists", "critic", "verifier"], dag=dag, debate=None)
    assert "debate_failed_open" in result.reasons
    assert result.score == 0.9


def test_debate_present_is_not_penalized():
    dag = _dag(nodes=[_node()])
    result = assess_confidence(["planner", "specialists", "critic", "verifier", "debate"], dag=dag, debate=_debate())
    assert "debate_failed_open" not in result.reasons
    assert result.score == 1.0


def test_no_verifier_in_team_means_no_debate_penalty_even_without_a_debate_result():
    # Debate is a standing tier-4 team member only when verifier ran --
    # absence of debate shouldn't be penalized outside tier 4.
    dag = _dag(nodes=[_node()])
    result = assess_confidence(["solver", "critic"], dag=dag, debate=None)
    assert "debate_failed_open" not in result.reasons
    assert result.score == 1.0


def test_unsupported_claims_are_penalized_scaled_by_ratio():
    dag = _dag(nodes=[_node()])
    evidence = EvidenceGraph(claims=[
        EvidenceClaim(claim="a", supported_by=["solve"], supported=True),
        EvidenceClaim(claim="b", supported_by=[], supported=False),
    ])
    result = assess_confidence(["solver", "critic", "evidence"], dag=dag, evidence=evidence)
    assert "unsupported_claims" in result.reasons
    assert result.score == 0.85  # 1.0 - 0.3 * (1/2)


def test_all_supported_claims_are_not_penalized():
    dag = _dag(nodes=[_node()])
    evidence = EvidenceGraph(claims=[EvidenceClaim(claim="a", supported_by=["solve"], supported=True)])
    result = assess_confidence(["solver", "critic", "evidence"], dag=dag, evidence=evidence)
    assert "unsupported_claims" not in result.reasons
    assert result.score == 1.0


def test_empty_claims_list_is_not_penalized():
    dag = _dag(nodes=[_node()])
    evidence = EvidenceGraph(claims=[])
    result = assess_confidence(["solver", "critic", "evidence"], dag=dag, evidence=evidence)
    assert "unsupported_claims" not in result.reasons
    assert result.score == 1.0


# --- stacking + clamping -------------------------------------------------------

def test_multiple_penalties_stack_and_clamp_to_zero():
    dag = _dag(status="partial", nodes=[_node(critique=CritiqueResult(satisfied=False))])
    verification = VerificationResult(satisfied=False)
    evidence = EvidenceGraph(claims=[EvidenceClaim(claim="a", supported_by=[], supported=False)])
    result = assess_confidence(
        ["planner", "specialists", "critic", "verifier", "evidence"],
        dag=dag, verification=verification, debate=None, evidence=evidence,
    )
    # 1.0 - 0.3 (dag) - 0.3 (critique) - 0.3 (verification) - 0.1 (debate) - 0.3 (claims) = -0.3 -> clamped to 0.0
    assert result.score == 0.0
    assert result.label == "low"
    assert set(result.reasons) == {
        "dag_partial_or_failed", "critique_unsatisfied", "verification_unsatisfied",
        "debate_failed_open", "unsupported_claims",
    }


# --- label boundaries -----------------------------------------------------------

def test_label_is_high_at_and_above_point_eight():
    dag = _dag(nodes=[_node()])
    result = assess_confidence(["solver", "critic"], dag=dag)
    assert result.score == 1.0
    assert result.label == "high"


def test_label_is_medium_between_point_five_and_point_eight():
    dag = _dag(nodes=[_node(critique=CritiqueResult(satisfied=False))])  # -0.3 -> 0.7
    result = assess_confidence(["solver", "critic"], dag=dag)
    assert result.score == 0.7
    assert result.label == "medium"


def test_label_is_low_below_point_five():
    dag = _dag(status="partial", nodes=[_node(critique=CritiqueResult(satisfied=False))])  # -0.3 -0.3 -> 0.4
    result = assess_confidence(["solver", "critic"], dag=dag)
    assert result.score == 0.4
    assert result.label == "low"
