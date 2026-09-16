import pytest

from app.agents.simulation import _build_scenario_request, run_simulations
from tests.helpers import build_test_engine


# --- pure request-building ----------------------------------------------------

def test_build_scenario_request_includes_task_and_scenario():
    req = _build_scenario_request("what's the capital of France?", "assume it's Lyon instead", None, None)
    joined = " ".join(m.content for m in req.messages if isinstance(m.content, str))
    assert "what's the capital of France?" in joined
    assert "assume it's Lyon instead" in joined


def test_build_scenario_request_includes_context_when_given():
    req = _build_scenario_request("task", "scenario", "the user is a beginner", None)
    joined = " ".join(m.content for m in req.messages if isinstance(m.content, str))
    assert "the user is a beginner" in joined


def test_build_scenario_request_omits_context_when_absent():
    req = _build_scenario_request("task", "scenario", None, None)
    joined = " ".join(m.content for m in req.messages if isinstance(m.content, str))
    assert "Context:" not in joined


# --- end-to-end run_simulations() via a real ChatEngine (FakeProvider-backed) -

@pytest.mark.asyncio
async def test_run_simulations_happy_path_with_two_scenarios(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"responses": [
        {"content": "under scenario A, the answer is X"},
        {"content": "under scenario B, the answer is Y"},
    ]}})

    runs = await run_simulations(engine, "task", ["scenario A", "scenario B"])

    assert len(runs) == 2
    assert runs[0].scenario == "scenario A"
    assert runs[0].answer == "under scenario A, the answer is X"
    assert runs[1].scenario == "scenario B"
    assert runs[1].answer == "under scenario B, the answer is Y"
    solo = engine.ctx.providers.get("solo")
    assert solo.call_count == 2


@pytest.mark.asyncio
async def test_run_simulations_truncates_past_the_max_scenario_cap(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"content": "an answer"}})

    runs = await run_simulations(engine, "task", ["a", "b", "c", "d"])

    # _MAX_SCENARIOS is 3 -- the 4th scenario ("d") is dropped entirely,
    # never even reaching the provider.
    assert [r.scenario for r in runs] == ["a", "b", "c"]
    solo = engine.ctx.providers.get("solo")
    assert solo.call_count == 3


@pytest.mark.asyncio
async def test_run_simulations_skips_blank_scenarios(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"content": "an answer"}})

    runs = await run_simulations(engine, "task", ["a real scenario", "  ", ""])

    assert [r.scenario for r in runs] == ["a real scenario"]
    solo = engine.ctx.providers.get("solo")
    assert solo.call_count == 1


@pytest.mark.asyncio
async def test_run_simulations_one_scenario_failing_still_returns_the_other(tmp_path):
    # A single failed candidate actually retries internally (RetryConfig's
    # default max_attempts=2 additional tries, i.e. 3 calls total) before
    # engine.handle_chat() gives up and raises NoAvailableModelError -- so
    # the first scenario's request must keep failing across all of those
    # attempts, not just once, to genuinely fail rather than succeed on a
    # retry.
    def _fail_for_first_three_calls(count: int) -> None:
        if count <= 3:
            from app.core.errors import ProviderServerError

            raise ProviderServerError("simulated 500", provider_id="solo")

    engine = await build_test_engine(tmp_path, {"solo": {
        "behavior": _fail_for_first_three_calls,
        "responses": [{"content": "the surviving scenario's answer"}],
    }})

    runs = await run_simulations(engine, "task", ["fails", "succeeds"])

    assert len(runs) == 1
    assert runs[0].scenario == "succeeds"
    assert runs[0].answer == "the surviving scenario's answer"


@pytest.mark.asyncio
async def test_run_simulations_returns_empty_list_when_every_scenario_fails(tmp_path):
    engine = await build_test_engine(tmp_path, {})  # no providers registered

    runs = await run_simulations(engine, "task", ["a", "b"])

    assert runs == []
