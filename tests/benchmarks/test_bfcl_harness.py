"""Tests for benchmarks/bfcl/run_benchmark.py's own deterministic logic:
schema conversion, tool_call decoding, and grading. These never touch a
live model -- the one end-to-end test mocks the HTTP boundary with respx,
the same pattern used throughout tests/providers/. Running the actual
benchmark against a real model is a separate, manual step (see
benchmarks/bfcl/README.md); it is not part of `pytest -q`."""
from __future__ import annotations

import json

import httpx
import respx

from benchmarks.bfcl.run_benchmark import (
    _to_json_schema,
    build_request,
    decode_tool_calls,
    grade_item,
    run_item,
)


def test_to_json_schema_converts_gorilla_dict_type_to_object():
    schema = {"type": "dict", "properties": {"base": {"type": "integer"}, "ratio": {"type": "float"}}}
    converted = _to_json_schema(schema)
    assert converted["type"] == "object"
    assert converted["properties"]["base"]["type"] == "integer"
    assert converted["properties"]["ratio"]["type"] == "number"


def test_to_json_schema_handles_nested_array_items():
    schema = {"type": "array", "items": {"type": "dict", "properties": {"x": {"type": "float"}}}}
    converted = _to_json_schema(schema)
    assert converted["type"] == "array"
    assert converted["items"]["type"] == "object"
    assert converted["items"]["properties"]["x"]["type"] == "number"


def test_build_request_uses_first_turn_and_wraps_functions_as_tools():
    item = {
        "id": "simple_python_0",
        "question": [[{"role": "user", "content": "hi"}]],
        "function": [{"name": "f", "parameters": {"type": "dict", "properties": {}}}],
    }
    body = build_request(item, "ollama/llama3.2:3b")
    assert body["model"] == "ollama/llama3.2:3b"
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["tools"][0]["type"] == "function"
    assert body["tools"][0]["function"]["parameters"]["type"] == "object"


def test_decode_tool_calls_parses_json_string_arguments():
    message = {"tool_calls": [{"function": {"name": "calculate_triangle_area", "arguments": json.dumps({"base": 10, "height": 5})}}]}
    decoded = decode_tool_calls(message)
    assert decoded == [{"calculate_triangle_area": {"base": 10, "height": 5}}]


def test_decode_tool_calls_returns_empty_list_when_no_tool_calls():
    assert decode_tool_calls({"content": "just text"}) == []


def test_decode_tool_calls_tolerates_malformed_arguments_json():
    message = {"tool_calls": [{"function": {"name": "f", "arguments": "{not json"}}]}
    decoded = decode_tool_calls(message)
    assert decoded == [{"f": {}}]


def test_grade_item_simple_correct():
    item = {
        "id": "simple_python_0",
        "function": [{"name": "calculate_triangle_area", "parameters": {"type": "dict", "properties": {"base": {"type": "integer"}, "height": {"type": "integer"}}, "required": ["base", "height"]}}],
        "ground_truth": [{"calculate_triangle_area": {"base": [10], "height": [5]}}],
    }
    decoded = [{"calculate_triangle_area": {"base": 10, "height": 5}}]
    verdict = grade_item("simple_python", item, decoded, "test-model")
    assert verdict["valid"] is True


def test_grade_item_simple_wrong_value():
    item = {
        "id": "simple_python_0",
        "function": [{"name": "calculate_triangle_area", "parameters": {"type": "dict", "properties": {"base": {"type": "integer"}, "height": {"type": "integer"}}, "required": ["base", "height"]}}],
        "ground_truth": [{"calculate_triangle_area": {"base": [10], "height": [5]}}],
    }
    decoded = [{"calculate_triangle_area": {"base": 99, "height": 5}}]
    verdict = grade_item("simple_python", item, decoded, "test-model")
    assert verdict["valid"] is False


def test_grade_item_irrelevance_correct_when_no_call_made():
    item = {"id": "irrelevance_0", "function": [{"name": "unrelated_fn"}], "ground_truth": None}
    verdict = grade_item("irrelevance", item, [], "test-model")
    assert verdict["valid"] is True


def test_grade_item_irrelevance_wrong_when_call_made():
    item = {"id": "irrelevance_0", "function": [{"name": "unrelated_fn"}], "ground_truth": None}
    verdict = grade_item("irrelevance", item, [{"unrelated_fn": {}}], "test-model")
    assert verdict["valid"] is False


@respx.mock
def test_run_item_end_to_end_against_a_mocked_openai_compatible_endpoint():
    item = {
        "id": "simple_python_0",
        "question": [[{"role": "user", "content": "area of a triangle, base 10, height 5"}]],
        "function": [{"name": "calculate_triangle_area", "parameters": {"type": "dict", "properties": {"base": {"type": "integer"}, "height": {"type": "integer"}}, "required": ["base", "height"]}}],
        "ground_truth": [{"calculate_triangle_area": {"base": [10], "height": [5]}}],
    }
    respx.post("http://localhost:20128/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [{"function": {"name": "calculate_triangle_area", "arguments": json.dumps({"base": 10, "height": 5})}}],
                        }
                    }
                ]
            },
        )
    )
    with httpx.Client(base_url="http://localhost:20128/v1") as client:
        result = run_item(client, "simple_python", item, "ollama/llama3.2:3b")
    assert result["valid"] is True
    assert result["model_output"] == [{"calculate_triangle_area": {"base": 10, "height": 5}}]


@respx.mock
def test_run_item_records_failure_without_raising_when_request_errors():
    item = {
        "id": "simple_python_1",
        "question": [[{"role": "user", "content": "..."}]],
        "function": [{"name": "f", "parameters": {"type": "dict", "properties": {}}}],
        "ground_truth": [{"f": {}}],
    }
    respx.post("http://localhost:20128/v1/chat/completions").mock(side_effect=httpx.ConnectError("refused"))
    with httpx.Client(base_url="http://localhost:20128/v1") as client:
        result = run_item(client, "simple_python", item, "ollama/llama3.2:3b")  # must not raise
    assert result["valid"] is False
    assert "request failed" in result["error"][0]
