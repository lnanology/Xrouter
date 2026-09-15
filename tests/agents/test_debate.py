import pytest

from app.agents.debate import (
    _build_advocate_request,
    _build_judge_request,
    _build_skeptic_request,
    debate,
)
from app.contracts.dag import DagNodeResult, DagRunResponse
from app.contracts.response import ChatCompletionChoice, ChatCompletionResponse
from tests.helpers import build_test_engine


def _node(id_: str, content: str = "x") -> DagNodeResult:
    response = ChatCompletionResponse(model="x", choices=[ChatCompletionChoice(message={"role": "assistant", "content": content})])
    return DagNodeResult(id=id_, status="success", response=response)


def _dag(nodes: list[DagNodeResult]) -> DagRunResponse:
    return DagRunResponse(id="dag_test", status="success", nodes=nodes, latency_ms=0.0)


# --- pure request-building ---------------------------------------------------

def test_build_advocate_request_includes_task_and_draft():
    dag = _dag([_node("a", "Paris")])
    req = _build_advocate_request("what's the capital of France?", "Paris", dag, None, None)
    joined = " ".join(m.content for m in req.messages if isinstance(m.content, str))
    assert "what's the capital of France?" in joined
    assert "Paris" in joined


def test_build_advocate_request_includes_context_when_given():
    dag = _dag([_node("a", "Paris")])
    req = _build_advocate_request("task", "draft", dag, "the user is a beginner", None)
    joined = " ".join(m.content for m in req.messages if isinstance(m.content, str))
    assert "the user is a beginner" in joined


def test_build_skeptic_request_includes_task_and_draft():
    dag = _dag([_node("a", "Paris")])
    req = _build_skeptic_request("what's the capital of France?", "Paris", dag, None, None)
    joined = " ".join(m.content for m in req.messages if isinstance(m.content, str))
    assert "what's the capital of France?" in joined
    assert "Paris" in joined


def test_build_judge_request_includes_draft_advocate_and_skeptic():
    req = _build_judge_request("task", "the draft", "the advocate's case", "the skeptic's case", None)
    joined = " ".join(m.content for m in req.messages if isinstance(m.content, str))
    assert "the draft" in joined
    assert "the advocate's case" in joined
    assert "the skeptic's case" in joined


def test_build_judge_request_truncates_long_arguments():
    long_argument = "x" * 10000
    req = _build_judge_request("task", "draft", long_argument, long_argument, None)
    joined = " ".join(m.content for m in req.messages if isinstance(m.content, str))
    assert len(joined) < 10000  # both arguments must have been capped, not embedded whole


# --- end-to-end debate() via a real ChatEngine (FakeProvider-backed) --------

@pytest.mark.asyncio
async def test_debate_happy_path_returns_the_judges_resolution(tmp_path):
    dag = _dag([_node("a", "Paris")])
    engine = await build_test_engine(tmp_path, {"solo": {"responses": [
        {"content": "the case for this answer"},
        {"content": "a counterpoint to consider"},
        {"content": "the final, strengthened answer"},
    ]}})

    answer, result = await debate(engine, "task", None, dag, "Paris is the capital")

    assert answer == "the final, strengthened answer"
    assert result is not None
    assert result.position == "Paris is the capital"
    assert result.advocate == "the case for this answer"
    assert result.skeptic == "a counterpoint to consider"
    assert result.resolution == "the final, strengthened answer"
    solo = engine.ctx.providers.get("solo")
    assert solo.call_count == 3


@pytest.mark.asyncio
async def test_debate_fails_open_when_no_provider_is_available(tmp_path):
    dag = _dag([_node("a")])
    engine = await build_test_engine(tmp_path, {})  # no providers registered

    answer, result = await debate(engine, "task", None, dag, "the original draft")

    assert answer == "the original draft"
    assert result is None


@pytest.mark.asyncio
async def test_debate_fails_open_when_a_later_call_fails(tmp_path):
    def _fail_from_second_call(count: int) -> None:
        if count >= 2:
            from app.core.errors import ProviderServerError

            raise ProviderServerError("simulated 500", provider_id="solo")

    dag = _dag([_node("a")])
    engine = await build_test_engine(tmp_path, {"solo": {
        "behavior": _fail_from_second_call,
        "responses": [{"content": "the case for this answer"}],
    }})

    answer, result = await debate(engine, "task", None, dag, "the original draft")

    # The advocate call succeeded, but the skeptic call failing must still
    # roll the whole debate back to the untouched draft -- a half-finished
    # debate is not a trustworthy revision.
    assert answer == "the original draft"
    assert result is None


@pytest.mark.asyncio
async def test_debate_fails_open_when_the_judge_returns_nothing_usable(tmp_path):
    dag = _dag([_node("a")])
    engine = await build_test_engine(tmp_path, {"solo": {"responses": [
        {"content": "the case for this answer"},
        {"content": "a counterpoint to consider"},
        {"content": ""},
    ]}})

    answer, result = await debate(engine, "task", None, dag, "the original draft")

    assert answer == "the original draft"
    assert result is None
