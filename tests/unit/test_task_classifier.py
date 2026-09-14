from app.contracts.request import ChatCompletionRequest, ChatMessage
from app.intelligence.task_classifier import TaskType, classify


def _req(text, tools=None):
    return ChatCompletionRequest(model="auto", messages=[ChatMessage(role="user", content=text)], tools=tools)


def test_trivial_chat_suggests_fastest():
    c = classify(_req("hi, how are you?"))
    assert c.task_type == TaskType.CHAT
    assert c.complexity <= 1
    assert c.suggested_policy == "fastest"


def test_code_keywords_detected():
    c = classify(_req("can you fix this bug in my python function? ```def foo(): pass```"))
    assert c.task_type == TaskType.CODE


def test_complex_code_task_suggests_quality():
    text = "please refactor this whole module step by step, design a better architecture, and debug the algorithm. " * 5
    c = classify(_req(text))
    assert c.task_type == TaskType.CODE
    assert c.complexity >= 2
    assert c.suggested_policy == "quality"


def test_research_keywords_detected():
    c = classify(_req("please research the latest news and cite your sources"))
    assert c.task_type == TaskType.RESEARCH
    assert c.suggested_policy == "reliable"


def test_reasoning_keywords_detected():
    c = classify(_req("prove that the square root of 2 is irrational, step by step"))
    assert c.task_type == TaskType.REASONING


def test_creative_keywords_detected():
    c = classify(_req("write a short story about a dragon, be creative"))
    assert c.task_type == TaskType.CREATIVE


def test_tools_present_forces_tool_use_type_and_reliable_policy():
    c = classify(_req("what's the weather", tools=[{"type": "function", "function": {"name": "get_weather"}}]))
    assert c.task_type == TaskType.TOOL_USE
    assert c.requires_tools is True
    assert c.suggested_policy == "reliable"


def test_very_hard_task_always_suggests_quality_regardless_of_type():
    text = "debug refactor architecture step by step algorithm optimi prove research compare and contrast " * 20
    c = classify(_req(text))
    assert c.complexity == 4
    assert c.suggested_policy == "quality"
