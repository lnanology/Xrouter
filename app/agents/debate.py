"""Debate (Phase 4, spec section 十九's own "Very hard" team table:
Planner + parallel specialists + Research + Debate + Verification +
Synthesizer). A genuine three-call adversarial-then-reconciled review of
the tier-4 draft answer app/agents/orchestrator.py's own
_synthesize_or_fallback already produced -- not a second Critic wearing a
different hat: an Advocate call argues the draft is correct and well-
supported, a Skeptic call argues the opposite (real gaps/overstatements,
not reflexive contrarianism), and a Judge call is shown both arguments
plus the original draft and writes ONE final, strengthened answer. That
Judge output is what a caller should actually use going forward -- Debate
revises the answer, it doesn't just annotate it.

Deliberately NOT wired between DAG execution and Verification the way
spec section 十九's left-to-right list might suggest -- app/execution/
plan_runner.py's run_plan_with_verification() is reused wholesale
(consistent with Phase 3's own reuse discipline), so Debate instead runs
on the DAG that already passed verification (or was re-planned until it
did). Debating a draft verification was about to throw away would waste
three calls on a losing draft; debating the winning one is what actually
matters.

Mandatory at tier 4 (app/agents/orchestrator.py, no opt-in flag -- Debate
is a spec-named standing member of that tier's team, same status the
Verifier already has there, not an invention of this codebase's own the
way Evidence Graph is), so it fails open with a sharper edge than
Evidence Graph's "just return nothing": any NoAvailableModelError from
the three calls, or an empty Judge output, returns the untouched draft
answer -- the same "an already-produced answer must never be lost to a
best-effort enrichment" discipline _synthesize_or_fallback itself already
applies to a failed synthesize() call."""
from __future__ import annotations

from typing import TYPE_CHECKING

from app.contracts.dag import DagRunResponse
from app.contracts.debate import DebateResult
from app.contracts.request import ChatCompletionRequest, ChatMessage
from app.contracts.response import extract_message_text
from app.core.errors import NoAvailableModelError
from app.execution.dag_summary import summarize_dag
from app.observability.logging import get_logger

if TYPE_CHECKING:
    from app.core.engine import ChatEngine

logger = get_logger("agents.debate")

# Keeps one side's own argument from crowding out the rest of the Judge's
# context -- the same context-crowding guard app/intelligence/critic.py's
# own _MAX_OUTPUT_CHARS applies to a single node's output.
_MAX_ARGUMENT_CHARS = 3000


def _build_case_user_message(task: str, draft: str, dag: DagRunResponse, context: str | None) -> str:
    parts = [f"Original task:\n{task}"]
    if context:
        parts.append(f"Context:\n{context}")
    parts.append(f"Draft answer:\n{draft}")
    parts.append(f"What each step produced:\n{summarize_dag(dag)}")
    return "\n\n".join(parts)


def _build_advocate_request(task: str, draft: str, dag: DagRunResponse, context: str | None, routing_policy: str | None) -> ChatCompletionRequest:
    system = (
        "You are the advocate stage of XRouter's Debate step. You will be "
        "shown the user's original task, a draft answer, and what each "
        "step of the run that produced it actually output. Make the "
        "strongest good-faith case that this draft answer is correct, "
        "complete, and well-supported by what the steps actually "
        "produced. Do not introduce new unsupported claims of your own."
    )
    messages = [ChatMessage(role="system", content=system), ChatMessage(role="user", content=_build_case_user_message(task, draft, dag, context))]
    return ChatCompletionRequest(model="auto", routing_policy=routing_policy, messages=messages)


def _build_skeptic_request(task: str, draft: str, dag: DagRunResponse, context: str | None, routing_policy: str | None) -> ChatCompletionRequest:
    system = (
        "You are the skeptic stage of XRouter's Debate step. You will be "
        "shown the user's original task, a draft answer, and what each "
        "step of the run that produced it actually output. Make the "
        "strongest good-faith case AGAINST this draft answer: what's "
        "missing, wrong, overstated, or unsupported by what the steps "
        "actually produced? Be specific and substantive, not reflexively "
        "contrarian -- if the draft genuinely has no real problems, say so."
    )
    messages = [ChatMessage(role="system", content=system), ChatMessage(role="user", content=_build_case_user_message(task, draft, dag, context))]
    return ChatCompletionRequest(model="auto", routing_policy=routing_policy, messages=messages)


def _build_judge_request(task: str, draft: str, advocate: str, skeptic: str, routing_policy: str | None) -> ChatCompletionRequest:
    system = (
        "You are the judge stage of XRouter's Debate step. You will be "
        "shown the user's original task, a draft answer, an advocate's "
        "case for it, and a skeptic's case against it. Write ONE final, "
        "strengthened answer for the user that fixes any real problems "
        "the skeptic raised while keeping what the advocate confirmed is "
        "solid. Answer directly, as if you had done it all yourself -- do "
        "not mention the debate, the advocate, or the skeptic."
    )
    user = (
        f"Original task:\n{task}\n\nDraft answer:\n{draft}\n\n"
        f"Advocate's case:\n{advocate[:_MAX_ARGUMENT_CHARS]}\n\n"
        f"Skeptic's case:\n{skeptic[:_MAX_ARGUMENT_CHARS]}"
    )
    messages = [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)]
    return ChatCompletionRequest(model="auto", routing_policy=routing_policy, messages=messages)


async def debate(
    engine: "ChatEngine", task: str, context: str | None, dag: DagRunResponse, draft_answer: str, routing_policy: str | None = None,
) -> tuple[str, DebateResult | None]:
    """Returns (final_answer, DebateResult | None). Fails open to
    (draft_answer, None) on any NoAvailableModelError, or when the
    Judge's own output comes back empty -- a perfectly good draft must
    never be replaced by nothing."""
    try:
        advocate_response = await engine.handle_chat(_build_advocate_request(task, draft_answer, dag, context, routing_policy))
        advocate_text = extract_message_text(advocate_response).strip()

        skeptic_response = await engine.handle_chat(_build_skeptic_request(task, draft_answer, dag, context, routing_policy))
        skeptic_text = extract_message_text(skeptic_response).strip()

        judge_response = await engine.handle_chat(_build_judge_request(task, draft_answer, advocate_text, skeptic_text, routing_policy))
        resolution_text = extract_message_text(judge_response).strip()
    except NoAvailableModelError as e:
        logger.warning("debate could not complete (%s); keeping the pre-debate draft answer", e)
        return draft_answer, None

    if not resolution_text:
        logger.warning("debate's judge stage returned no usable answer; keeping the pre-debate draft answer")
        return draft_answer, None

    result = DebateResult(position=draft_answer, advocate=advocate_text, skeptic=skeptic_text, resolution=resolution_text)
    return resolution_text, result
