from app.contracts.request import ChatCompletionRequest, ChatMessage
from app.routing.classifier import classify_complexity


def test_trivial_short_prompt_is_low_complexity():
    req = ChatCompletionRequest(model="auto", messages=[ChatMessage(role="user", content="hi")])
    assert classify_complexity(req) <= 1


def test_long_prompt_with_hard_keywords_is_high_complexity():
    text = "Please design a distributed system architecture, step by step, and prove its correctness. " * 5
    req = ChatCompletionRequest(model="auto", messages=[ChatMessage(role="user", content=text)])
    assert classify_complexity(req) >= 3


def test_complexity_bounded_at_4():
    text = "debug refactor architecture step by step algorithm optimi prove research compare and contrast " * 20
    req = ChatCompletionRequest(
        model="auto",
        messages=[ChatMessage(role="user", content=text)] * 8,
        tools=[{"type": "function", "function": {"name": "x"}}],
    )
    assert classify_complexity(req) == 4
