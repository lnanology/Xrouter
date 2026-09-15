"""Evidence Graph (Phase 4, spec 三十六 -- the first of the five Phase 4
pieces; see README's Phase 4 section for the rest, not yet built).

Traces which of a finished Orchestrator run's own claims are actually
backed by which DAG step, so a caller can tell "this came from step X's
own output" apart from "the model just asserted this from its own
knowledge, nothing here checked it" -- distinct from the Critic (judges
one step's own output against its own instruction) and the Verifier
(judges the whole run against the original task, once): this instead
looks at the *finished answer itself*, claim by claim, and traces each
one back to whichever step (if any) actually produced it.

Despite the name, this ships as a flat claim -> DAG-node-id mapping, not
a literal multi-hop graph -- nothing in XRouter today produces or
consumes multi-hop evidence relationships, and building traversal
machinery nobody uses would be exactly the "不要新增不必要的
infrastructure" (spec rule 15) this project keeps refusing to do. It's
also deliberately DAG-node-level, not URL-level: app/agents/
researcher.py's web_search results reach the model as prose the system
prompt asks it to cite inline, but nothing in app/execution/tool_loop.py
captures those citations as structured, addressable objects yet -- so an
"evidence graph" that pretended to trace claims to specific URLs today
would be faking data XRouter doesn't actually have (spec rule 4). DAG
node outputs are the one thing this system genuinely already has in
structured form, so that's what v1 traces against; a URL-level layer can
extend this later, the same "swappable, add without breaking callers"
shape app/retrieval/'s Retriever Protocol already uses for the same
reason.

Same "no fake placeholder functionality" principle as the Planner/
Verifier/Critic: an actual LLM judgment (a forced "submit_evidence_graph"
tool call), never a heuristic. Same fail-open philosophy too: a missing/
malformed tool call, or NoAvailableModelError from the call itself,
returns an empty EvidenceGraph rather than blocking or retrying an
already-produced answer -- this is advisory metadata about an answer
XRouter already committed to returning, never something that should hold
it hostage."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from app.contracts.dag import DagRunResponse
from app.contracts.evidence import EvidenceClaim, EvidenceGraph
from app.contracts.request import ChatCompletionRequest, ChatMessage
from app.contracts.response import ChatCompletionResponse
from app.core.errors import NoAvailableModelError
from app.execution.dag_summary import summarize_dag
from app.observability.logging import get_logger

if TYPE_CHECKING:
    from app.core.engine import ChatEngine

logger = get_logger("intelligence.evidence")

EVIDENCE_TOOL_NAME = "submit_evidence_graph"

_MODEL_KNOWLEDGE = "model_knowledge"


def _build_evidence_tool_schema(node_ids: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": EVIDENCE_TOOL_NAME,
            "description": "Report each concrete claim in the final answer and which step (if any) actually backs it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "claims": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "claim": {"type": "string", "description": "One concrete claim/assertion from the final answer."},
                                "supported_by": {
                                    "type": "array",
                                    "items": {"type": "string", "enum": [*node_ids, _MODEL_KNOWLEDGE]},
                                    "description": "Which step(s) actually produced this claim, or 'model_knowledge' if none did.",
                                },
                                "supported": {
                                    "type": "boolean",
                                    "description": "true if what supported_by cites genuinely backs this claim, false if it's asserted without real support.",
                                },
                            },
                            "required": ["claim", "supported_by", "supported"],
                        },
                    },
                },
                "required": ["claims"],
            },
        },
    }


def _build_evidence_request(
    task: str, answer: str, dag: DagRunResponse, routing_policy: str | None,
) -> ChatCompletionRequest:
    node_ids = [n.id for n in dag.nodes]
    system = (
        "You are the evidence-tracing stage of XRouter, an AI gateway. You "
        "will be shown the user's original task, the final answer that was "
        "given, and what each step of the run that produced it actually "
        "output. List the concrete claims the final answer makes, and for "
        "each one, cite which step's own output it actually came from -- or "
        "'model_knowledge' if it isn't traceable to any step. Call "
        f"{EVIDENCE_TOOL_NAME} with your findings -- do not answer the task "
        "yourself."
    )
    user = f"Original task:\n{task}\n\nFinal answer:\n{answer}\n\nWhat each step produced:\n{summarize_dag(dag)}"
    messages = [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)]
    return ChatCompletionRequest(
        model="auto", messages=messages, routing_policy=routing_policy,
        tools=[_build_evidence_tool_schema(node_ids)],
        tool_choice={"type": "function", "function": {"name": EVIDENCE_TOOL_NAME}},
    )


def _extract_tool_arguments(response: ChatCompletionResponse) -> str | None:
    if not response.choices:
        return None
    message = response.choices[0].message
    tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
    if not tool_calls:
        return None
    fn = tool_calls[0].get("function", {}) if isinstance(tool_calls[0], dict) else {}
    return fn.get("arguments")


def _sanitize_claims(raw_claims: list[dict], valid_ids: set[str]) -> list[EvidenceClaim]:
    claims: list[EvidenceClaim] = []
    for raw in raw_claims:
        try:
            claim_text = raw["claim"]
            supported = bool(raw.get("supported", True))
        except (KeyError, TypeError):
            continue
        # Never trust the enum was actually honored -- drop any cited id
        # that isn't a real node id or "model_knowledge", the same
        # defense-in-depth app/intelligence/planner.py's
        # _validate_plan_shape applies to a hallucinated enable_tools
        # entry, just without a retry: a claim ending up with fewer (or
        # zero) supported_by ids simply looks less-supported, which is
        # already an honest outcome for a hallucinated citation.
        supported_by = [i for i in (raw.get("supported_by") or []) if i in valid_ids]
        claims.append(EvidenceClaim(claim=claim_text, supported_by=supported_by, supported=supported))
    return claims


async def build_evidence_graph(
    engine: "ChatEngine", task: str, answer: str, dag: DagRunResponse, routing_policy: str | None = None,
) -> EvidenceGraph:
    request = _build_evidence_request(task, answer, dag, routing_policy)
    try:
        response = await engine.handle_chat(request)
    except NoAvailableModelError as e:
        logger.warning("evidence tracing could not get an answer (%s); returning an empty evidence graph", e)
        return EvidenceGraph(claims=[])

    raw = _extract_tool_arguments(response)
    if raw is None:
        logger.warning("evidence tracing did not call %s; returning an empty evidence graph", EVIDENCE_TOOL_NAME)
        return EvidenceGraph(claims=[])

    try:
        data = json.loads(raw)
        raw_claims = data["claims"]
        if not isinstance(raw_claims, list):
            raise TypeError("claims was not a list")
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        logger.warning("evidence tracing returned an unusable %s call (%s); returning an empty evidence graph", EVIDENCE_TOOL_NAME, e)
        return EvidenceGraph(claims=[])

    valid_ids = {n.id for n in dag.nodes} | {_MODEL_KNOWLEDGE}
    return EvidenceGraph(claims=_sanitize_claims(raw_claims, valid_ids))
