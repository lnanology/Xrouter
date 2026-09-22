#!/usr/bin/env python3
"""Runs a real BFCL (Berkeley Function-Calling Leaderboard) subset
against a running XRouter server's own OpenAI-compatible
/v1/chat/completions endpoint, and grades the results with BFCL's own
vendored AST checker (see vendor/NOTICE).

This is a standalone credibility tool, not part of the shipped server --
run it by hand against a live `scripts/start.sh` instance:

    python3 benchmarks/bfcl/run_benchmark.py --model ollama/llama3.2:3b

See benchmarks/bfcl/README.md for the full methodology, category list,
and how the resulting score gets into the main README.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for `benchmarks.bfcl.vendor` imports

from benchmarks.bfcl.vendor.ast_checker import ast_checker  # noqa: E402
from benchmarks.bfcl.vendor.constants import Language  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"

ALL_CATEGORIES = ["simple_python", "multiple", "parallel", "parallel_multiple", "irrelevance"]

# BFCL's own type vocabulary -> real JSON Schema, reproduced from
# bfcl_eval/constants/type_mappings.py's GORILLA_TO_OPENAPI (see vendor/NOTICE
# for why this is reproduced inline rather than importing that module).
_GORILLA_TO_OPENAPI = {
    "integer": "integer", "number": "number", "float": "number", "string": "string",
    "boolean": "boolean", "bool": "boolean", "array": "array", "list": "array",
    "dict": "object", "object": "object", "tuple": "array", "any": "string",
    "byte": "integer", "short": "integer", "long": "integer", "double": "number",
    "char": "string", "ArrayList": "array", "Array": "array", "HashMap": "object",
    "Hashtable": "object", "Queue": "array", "Stack": "array", "Any": "string",
    "String": "string", "Bigint": "integer",
}


def _to_json_schema(node: Any) -> Any:
    """Recursively rewrites BFCL's type vocabulary (e.g. "type": "dict")
    into real JSON Schema (e.g. "type": "object") inside a function's
    `parameters` block, so the schema we hand to a real OpenAI-compatible
    endpoint's `tools` field is actually valid JSON Schema."""
    if isinstance(node, dict):
        out = {k: _to_json_schema(v) for k, v in node.items()}
        if "type" in out and isinstance(out["type"], str):
            out["type"] = _GORILLA_TO_OPENAPI.get(out["type"], out["type"])
        return out
    if isinstance(node, list):
        return [_to_json_schema(v) for v in node]
    return node


def load_category(category: str, limit: int | None) -> list[dict]:
    """Joins a category's question file with its possible_answer file (if
    any -- irrelevance has none, see BFCL's own eval_runner.py) by id."""
    questions = []
    with (DATA_DIR / f"BFCL_v4_{category}.json").open() as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    if limit is not None:
        questions = questions[:limit]

    answers: dict[str, Any] = {}
    answer_path = DATA_DIR / "possible_answer" / f"BFCL_v4_{category}.json"
    if answer_path.exists():
        with answer_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    answers[row["id"]] = row["ground_truth"]

    for q in questions:
        q["ground_truth"] = answers.get(q["id"])
    return questions


def build_request(item: dict, model: str) -> dict:
    messages = list(item["question"][0])  # single-turn categories: one turn
    tools = [{"type": "function", "function": _to_json_schema(fn)} for fn in item["function"]]
    return {"model": model, "messages": messages, "tools": tools, "tool_choice": "auto"}


def decode_tool_calls(message: dict) -> list[dict]:
    """OpenAI-style tool_calls -> BFCL's expected [{func_name: {param: value}}, ...]."""
    decoded = []
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        name = fn.get("name")
        raw_args = fn.get("arguments")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
        except json.JSONDecodeError:
            args = {}
        if name:
            decoded.append({name: args})
    return decoded


def grade_item(category: str, item: dict, decoded: list[dict], model_name: str) -> dict:
    if category == "irrelevance":
        valid = len(decoded) == 0
        return {"valid": valid, "error": [] if valid else ["Model called a function when none applied."]}
    return ast_checker(item["function"], decoded, item["ground_truth"], Language.PYTHON, category, model_name)


def run_item(client: httpx.Client, category: str, item: dict, model: str) -> dict:
    body = build_request(item, model)
    started = time.time()
    try:
        resp = client.post("/chat/completions", json=body, timeout=300.0)
        resp.raise_for_status()
        message = resp.json()["choices"][0]["message"]
    except Exception as e:  # a single item failing must never abort the whole run
        return {
            "id": item["id"], "category": category, "valid": False,
            "error": [f"request failed: {e}"], "latency_ms": (time.time() - started) * 1000,
        }

    decoded = decode_tool_calls(message)
    verdict = grade_item(category, item, decoded, model)
    return {
        "id": item["id"], "category": category, "valid": bool(verdict.get("valid")),
        "error": verdict.get("error", []), "error_type": verdict.get("error_type"),
        "model_output": decoded, "latency_ms": (time.time() - started) * 1000,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:20128/v1")
    parser.add_argument("--model", required=True, help="e.g. ollama/llama3.2:3b")
    parser.add_argument("--categories", nargs="+", default=ALL_CATEGORIES, choices=ALL_CATEGORIES)
    parser.add_argument("--limit", type=int, default=None, help="cap items per category (smoke-test runs)")
    parser.add_argument("--out", default=None, help="results JSON path (default: results_<timestamp>.json)")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else Path(__file__).parent / f"results_{int(time.time())}.json"
    all_results: list[dict] = []

    with httpx.Client(base_url=args.base_url) as client:
        for category in args.categories:
            items = load_category(category, args.limit)
            print(f"== {category}: {len(items)} items ==")
            for i, item in enumerate(items, 1):
                result = run_item(client, category, item, args.model)
                all_results.append(result)
                status = "PASS" if result["valid"] else "FAIL"
                print(f"  [{i}/{len(items)}] {result['id']}: {status}")

    summary: dict[str, dict] = {}
    for category in args.categories:
        cat_results = [r for r in all_results if r["category"] == category]
        n = len(cat_results)
        passed = sum(1 for r in cat_results if r["valid"])
        summary[category] = {"n": n, "passed": passed, "accuracy": (passed / n) if n else 0.0}

    overall_n = len(all_results)
    overall_passed = sum(1 for r in all_results if r["valid"])
    summary["overall"] = {
        "n": overall_n, "passed": overall_passed,
        "accuracy": (overall_passed / overall_n) if overall_n else 0.0,
    }

    out_path.write_text(json.dumps({"model": args.model, "categories": args.categories, "summary": summary, "items": all_results}, indent=2))

    print("\n| Category | N | Accuracy |")
    print("|---|---|---|")
    for category in args.categories:
        s = summary[category]
        print(f"| {category} | {s['n']} | {s['accuracy']:.1%} |")
    print(f"| **Overall** | {summary['overall']['n']} | **{summary['overall']['accuracy']:.1%}** |")
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
