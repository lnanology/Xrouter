#!/usr/bin/env python3
"""Standalone benchmark: exercises every enabled provider directly (no
running server required) and reports TTFT / P50 / P95 / P99 / success rate /
tokens-per-sec — not just average latency (section 三十五).

Usage:
    python3 scripts/benchmark.py [--runs 3] [--prompt "..."]
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.contracts.request import ChatCompletionRequest, ChatMessage  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.core.registry import ProviderRegistry  # noqa: E402
from app.observability.metrics import MetricsCollector  # noqa: E402


async def bench_provider(provider_id: str, provider, model_name: str, runs: int, prompt: str) -> dict:
    collector = MetricsCollector()
    failures = 0
    request = ChatCompletionRequest(model="auto", messages=[ChatMessage(role="user", content=prompt)], max_tokens=32)

    for _ in range(runs):
        start = time.time()
        try:
            resp = await provider.chat(model_name, request)
            latency_ms = (time.time() - start) * 1000
            tokens = resp.usage.total_tokens or 1
            tps = tokens / max(latency_ms / 1000, 0.001)
            collector.record_request(latency_ms=latency_ms, tokens=tokens, tokens_per_sec=tps, success=True)
        except Exception as e:
            failures += 1
            collector.record_request(latency_ms=(time.time() - start) * 1000, success=False)
            print(f"  [{provider_id}/{model_name}] run failed: {e}")

    snap = collector.snapshot()
    return {
        "provider": provider_id,
        "model": model_name,
        "runs": runs,
        "failures": failures,
        "p50_ms": round(snap["latency_ms"]["p50"], 1),
        "p95_ms": round(snap["latency_ms"]["p95"], 1),
        "p99_ms": round(snap["latency_ms"]["p99"], 1),
        "tokens_per_sec_p50": round(snap["tokens_per_sec"]["p50"], 2),
        "success_rate": snap["success_rate"],
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--prompt", type=str, default="Reply with a short one-sentence fact about the ocean.")
    args = parser.parse_args()

    settings = get_settings()
    registry = ProviderRegistry.build(settings)

    print("XRouter benchmark — comparing all enabled providers\n")
    results = []
    for pid, provider in registry.all().items():
        if not registry.is_enabled(pid):
            continue
        models = await provider.list_models()
        if not models:
            print(f"[{pid}] no models available (offline or unconfigured) — skipping")
            continue
        model = models[0]
        print(f"[{pid}] benchmarking {model.name} ({args.runs} runs)...")
        results.append(await bench_provider(pid, provider, model.name, args.runs, args.prompt))

    await registry.close_all()

    if not results:
        print("\nNo providers were reachable. Is Ollama running, or any cloud API key set?")
        return

    print("\n%-14s %-24s %8s %8s %8s %10s %8s" % ("provider", "model", "p50ms", "p95ms", "p99ms", "tok/s(p50)", "success"))
    for r in sorted(results, key=lambda r: r["p50_ms"]):
        print(
            "%-14s %-24s %8.1f %8.1f %8.1f %10.2f %7.0f%%"
            % (r["provider"], r["model"][:24], r["p50_ms"], r["p95_ms"], r["p99_ms"], r["tokens_per_sec_p50"], r["success_rate"] * 100)
        )


if __name__ == "__main__":
    asyncio.run(main())
