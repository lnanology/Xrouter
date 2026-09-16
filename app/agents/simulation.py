"""Simulation (Phase 4, spec 三十六 -- the fourth of the five Phase 4
pieces; see README's Phase 4 section for the rest).

Given the original task and one or more caller-supplied "changed premise"
scenarios, actually re-answers the task once per scenario, under that
changed premise -- a genuine re-execution, not a guess about what would
happen. That's the deliberate division of labor with Counterfactual
(Phase 4's third piece, app/intelligence/counterfactual.py): Counterfactual
identifies which of an answer's own assumptions are load-bearing and
*guesses* how the answer would change if one didn't hold; Simulation is
what a caller reaches for once they have a concrete premise in hand and
want to know what XRouter *actually* says under it. XRouter itself never
invents the scenarios here -- doing so would either duplicate
Counterfactual's own judgment or silently second-guess it, and the spec's
"不要新增不必要的infrastructure" (rule 15) cuts against building the same
feature twice under two names.

Each scenario is answered by a single plain engine.handle_chat() prose
call (same family as app/agents/synthesizer.py/debate.py -- a generative
answer, not a structured judgment, so no forced tool call), never a
recursive call back into app/agents/orchestrator.py's own orchestrate().
Re-running the whole Planner/Critic/Verifier/Debate pipeline per scenario
would make a `simulate` list an unbounded cost multiplier on an already-
expensive tier-4 request; one direct call per scenario keeps this
honestly scoped. Bounded at _MAX_SCENARIOS -- extra entries are dropped
with a logged warning rather than silently fanning out further.

Fail-open per scenario, inside this module: scenarios are independent, so
one NoAvailableModelError only drops that one scenario's result, never
the whole batch -- same "fail open, keep going" discipline Evidence Graph
and Counterfactual use, not Debate's "roll the whole thing back" (there's
no single draft to protect here) or Synthesizer's "propagate to a caller
with a real fallback" (there's no meaningful fallback content for one
simulated scenario -- it simply doesn't appear in the results)."""
from __future__ import annotations

from typing import TYPE_CHECKING

from app.contracts.request import ChatCompletionRequest, ChatMessage
from app.contracts.response import extract_message_text
from app.contracts.simulation import SimulationRun
from app.core.errors import NoAvailableModelError
from app.observability.logging import get_logger

if TYPE_CHECKING:
    from app.core.engine import ChatEngine

logger = get_logger("agents.simulation")

_MAX_SCENARIOS = 3


def _build_scenario_request(
    task: str, scenario: str, context: str | None, routing_policy: str | None,
) -> ChatCompletionRequest:
    system = (
        "You are the simulation stage of XRouter, an AI gateway. You will "
        "be shown the user's original task and a changed premise. Actually "
        "work out what the answer to the task genuinely becomes under that "
        "changed premise -- answer it for real, as if the premise were "
        "true, rather than describing how the answer might change."
    )
    parts = [f"Original task:\n{task}", f"Changed premise for this simulation:\n{scenario}"]
    if context:
        parts.append(f"Context:\n{context}")
    messages = [ChatMessage(role="system", content=system), ChatMessage(role="user", content="\n\n".join(parts))]
    return ChatCompletionRequest(model="auto", routing_policy=routing_policy, messages=messages)


async def run_simulations(
    engine: "ChatEngine", task: str, scenarios: list[str], context: str | None = None, routing_policy: str | None = None,
) -> list[SimulationRun]:
    usable = [s for s in scenarios if s and s.strip()]
    if len(usable) > _MAX_SCENARIOS:
        logger.warning("simulate had %d scenarios, more than the cap of %d; dropping the rest", len(usable), _MAX_SCENARIOS)
        usable = usable[:_MAX_SCENARIOS]

    runs: list[SimulationRun] = []
    for scenario in usable:
        request = _build_scenario_request(task, scenario, context, routing_policy)
        try:
            response = await engine.handle_chat(request)
        except NoAvailableModelError as e:
            logger.warning("simulation scenario %r could not get an answer (%s); dropping it from the results", scenario, e)
            continue
        answer = extract_message_text(response).strip()
        if not answer:
            logger.warning("simulation scenario %r returned no usable answer; dropping it from the results", scenario)
            continue
        runs.append(SimulationRun(scenario=scenario, answer=answer))
    return runs
