from app.contracts.request import ChatCompletionRequest, ChatMessage
from app.contracts.response import ChatCompletionChoice, ChatCompletionResponse, Usage
from app.intelligence.quality_gate import assess


def _req(tool_choice=None):
    return ChatCompletionRequest(model="auto", messages=[ChatMessage(role="user", content="hi")], tool_choice=tool_choice)


def _resp(content, finish_reason="stop", tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return ChatCompletionResponse(
        model="p/m",
        choices=[ChatCompletionChoice(index=0, message=message, finish_reason=finish_reason)],
        usage=Usage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
    )


def test_normal_response_passes():
    a = assess(_resp("The capital of France is Paris."), _req())
    assert a.passed is True
    assert a.score == 1.0
    assert a.reasons == []


def test_empty_response_fails():
    a = assess(_resp(""), _req())
    assert a.passed is False
    assert "empty_response" in a.reasons


def test_whitespace_only_response_fails():
    a = assess(_resp("   \n  "), _req())
    assert a.passed is False
    assert "empty_response" in a.reasons


def test_truncated_output_flagged():
    a = assess(_resp("this got cut off because max_tokens", finish_reason="length"), _req())
    assert "truncated_output" in a.reasons
    assert a.score < 1.0


def test_degenerate_word_repetition_fails():
    text = "the the the the the the the the the the"
    a = assess(_resp(text), _req())
    assert a.passed is False
    assert "degenerate_word_repetition" in a.reasons


def test_degenerate_char_repetition_fails():
    text = "!" * 40
    a = assess(_resp(text), _req())
    assert a.passed is False
    assert "degenerate_char_repetition" in a.reasons


def test_short_reply_not_penalized_for_repetition():
    # Too short for the repetition heuristics to fire — a legitimately
    # short, correct answer must never be flagged just for being short.
    a = assess(_resp("Yes."), _req())
    assert a.passed is True


def test_forced_tool_call_missing_fails():
    a = assess(_resp("I can help with that."), _req(tool_choice={"type": "function", "function": {"name": "get_weather"}}))
    assert a.passed is False
    assert "missing_forced_tool_call" in a.reasons


def test_forced_tool_call_present_passes():
    a = assess(
        _resp("", tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}]),
        _req(tool_choice={"type": "function", "function": {"name": "get_weather"}}),
    )
    assert a.passed is True


def test_unforced_tool_choice_does_not_require_tool_call():
    a = assess(_resp("Here's a plain text answer."), _req(tool_choice="auto"))
    assert a.passed is True


def test_custom_min_score_threshold():
    # A truncated-but-otherwise-fine response scores 0.7 — passes at the
    # default 0.5 threshold but can be made to fail with a stricter one.
    resp = _resp("a reasonably long and complete-looking answer", finish_reason="length")
    assert assess(resp, _req(), min_score=0.5).passed is True
    assert assess(resp, _req(), min_score=0.8).passed is False
