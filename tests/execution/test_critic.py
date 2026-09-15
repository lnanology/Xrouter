import json

import pytest

from app.contracts.request import ChatCompletionRequest, ChatMessage
from app.contracts.response import ChatCompletionChoice, ChatCompletionResponse
from app.execution.critique_loop import run_with_critique
from app.intelligence.critic import CRITIQUE_TOOL_NAME, _build_critique_request, critique
from tests.helpers import build_test_engine


def _critique_tool_call(satisfied: bool, feedback: str = "") -> dict:
    return {
        "id": "call_1", "type": "function",
        "function": {"name": CRITIQUE_TOOL_NAME, "arguments": json.dumps({"satisfied": satisfied, "feedback": feedback})},
    }


def _response(content: str) -> ChatCompletionResponse:
    return ChatCompletionResponse(model="x", choices=[ChatCompletionChoice(message={"role": "assistant", "content": content})])


# --- pure ---------------------------------------------------------------------

def test_build_critique_request_forces_the_submit_critique_tool():
    req = _build_critique_request("translate hello to Spanish", "hola", None)
    assert req.tool_choice == {"type": "function", "function": {"name": CRITIQUE_TOOL_NAME}}
    joined = " ".join(m.content for m in req.messages if isinstance(m.content, str))
    assert "translate hello to Spanish" in joined
    assert "hola" in joined


# --- end-to-end critique() via a real ChatEngine (FakeProvider-backed) -------

@pytest.mark.asyncio
async def test_critique_reports_satisfied(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"tool_calls": [_critique_tool_call(True)]}})
    result = await critique(engine, "say hi", _response("hi"))
    assert result.satisfied is True
    assert result.feedback is None


@pytest.mark.asyncio
async def test_critique_reports_unsatisfied_with_feedback(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"tool_calls": [_critique_tool_call(False, "too vague")]}})
    result = await critique(engine, "give a detailed answer", _response("uh, stuff"))
    assert result.satisfied is False
    assert result.feedback == "too vague"


@pytest.mark.asyncio
async def test_critique_fails_open_when_no_tool_call_returned(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"content": "looks fine to me"}})
    result = await critique(engine, "say hi", _response("hi"))
    assert result.satisfied is True


@pytest.mark.asyncio
async def test_critique_fails_open_on_malformed_tool_arguments(tmp_path):
    bad_call = {"id": "call_1", "type": "function", "function": {"name": CRITIQUE_TOOL_NAME, "arguments": "not json"}}
    engine = await build_test_engine(tmp_path, {"solo": {"tool_calls": [bad_call]}})
    result = await critique(engine, "say hi", _response("hi"))
    assert result.satisfied is True


@pytest.mark.asyncio
async def test_critique_fails_open_when_no_provider_available(tmp_path):
    engine = await build_test_engine(tmp_path, {})  # no providers registered at all
    result = await critique(engine, "say hi", _response("hi"))
    assert result.satisfied is True


# --- run_with_critique (the retry loop) ---------------------------------------

def _node_request(content="write a haiku about the sea"):
    return ChatCompletionRequest(model="auto", messages=[ChatMessage(role="user", content=content)])


@pytest.mark.asyncio
async def test_run_with_critique_satisfied_first_try_never_calls_responder_again(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"tool_calls": [_critique_tool_call(True)]}})
    call_count = 0

    async def responder(req):
        nonlocal call_count
        call_count += 1
        return _response("a haiku")

    response, result = await run_with_critique(engine, _node_request(), _response("a haiku"), responder)

    assert result.satisfied is True
    assert call_count == 0  # the original response was already accepted, never re-run
    assert response.choices[0].message["content"] == "a haiku"


@pytest.mark.asyncio
async def test_run_with_critique_unsatisfied_then_satisfied_retries_once(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"responses": [
        {"tool_calls": [_critique_tool_call(False, "not a haiku, wrong syllable count")]},
        {"tool_calls": [_critique_tool_call(True)]},
    ]}})
    responder_calls: list[ChatCompletionRequest] = []

    async def responder(req):
        responder_calls.append(req)
        return _response("a corrected haiku")

    response, result = await run_with_critique(engine, _node_request(), _response("not quite a haiku"), responder, max_retries=1)

    assert result.satisfied is True
    assert len(responder_calls) == 1
    assert response.choices[0].message["content"] == "a corrected haiku"
    # the feedback must actually reach the re-run request
    joined = " ".join(m.content for m in responder_calls[0].messages if isinstance(m.content, str))
    assert "wrong syllable count" in joined


@pytest.mark.asyncio
async def test_run_with_critique_gives_up_after_exhausting_retries(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"tool_calls": [_critique_tool_call(False, "still wrong")]}})
    responder_calls: list[ChatCompletionRequest] = []

    async def responder(req):
        responder_calls.append(req)
        return _response("still not quite right")

    response, result = await run_with_critique(engine, _node_request(), _response("first attempt"), responder, max_retries=2)

    assert result.satisfied is False
    assert len(responder_calls) == 2  # exactly max_retries re-runs, then it stops
    assert response.choices[0].message["content"] == "still not quite right"


@pytest.mark.asyncio
async def test_run_with_critique_zero_max_retries_never_reruns(tmp_path):
    engine = await build_test_engine(tmp_path, {"solo": {"tool_calls": [_critique_tool_call(False, "wrong")]}})
    call_count = 0

    async def responder(req):
        nonlocal call_count
        call_count += 1
        return _response("x")

    response, result = await run_with_critique(engine, _node_request(), _response("x"), responder, max_retries=0)

    assert result.satisfied is False
    assert call_count == 0
