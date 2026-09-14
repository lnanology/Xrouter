"""In-process metrics collector. Deliberately not average-only: tracks
latency distributions so P50/P95/P99 are available, per section 二十四."""
from __future__ import annotations

import bisect
import threading
from dataclasses import dataclass, field


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * pct
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


@dataclass
class _Series:
    values: list[float] = field(default_factory=list)
    max_samples: int = 2000

    def add(self, v: float) -> None:
        bisect.insort(self.values, v)
        if len(self.values) > self.max_samples:
            # Drop an arbitrary (not necessarily oldest) sample to bound memory;
            # fine for a rolling distributional estimate.
            self.values.pop(0)

    def percentiles(self) -> dict[str, float]:
        return {"p50": _percentile(self.values, 0.50), "p95": _percentile(self.values, 0.95), "p99": _percentile(self.values, 0.99)}


class MetricsCollector:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latency = _Series()
        self._ttft = _Series()
        self._tokens_per_sec = _Series()
        self.counters: dict[str, int] = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_timeout": 0,
            "requests_429": 0,
            "requests_failed": 0,
            "fallback_triggered": 0,
            "cache_hit": 0,
            "cache_miss": 0,
        }
        self._tokens_total = 0

    def record_request(
        self,
        *,
        latency_ms: float | None = None,
        ttft_ms: float | None = None,
        tokens: int = 0,
        tokens_per_sec: float | None = None,
        success: bool = True,
        timeout: bool = False,
        rate_limited: bool = False,
        fallback: bool = False,
        cache_hit: bool | None = None,
    ) -> None:
        with self._lock:
            self.counters["requests_total"] += 1
            if success:
                self.counters["requests_success"] += 1
            else:
                self.counters["requests_failed"] += 1
            if timeout:
                self.counters["requests_timeout"] += 1
            if rate_limited:
                self.counters["requests_429"] += 1
            if fallback:
                self.counters["fallback_triggered"] += 1
            if cache_hit is True:
                self.counters["cache_hit"] += 1
            elif cache_hit is False:
                self.counters["cache_miss"] += 1
            if latency_ms is not None:
                self._latency.add(latency_ms)
            if ttft_ms is not None:
                self._ttft.add(ttft_ms)
            if tokens_per_sec is not None:
                self._tokens_per_sec.add(tokens_per_sec)
            self._tokens_total += tokens

    def snapshot(self) -> dict:
        with self._lock:
            total = max(self.counters["requests_total"], 1)
            return {
                "counters": dict(self.counters),
                "latency_ms": self._latency.percentiles(),
                "ttft_ms": self._ttft.percentiles(),
                "tokens_per_sec": self._tokens_per_sec.percentiles(),
                "success_rate": self.counters["requests_success"] / total,
                "timeout_rate": self.counters["requests_timeout"] / total,
                "rate_limit_rate": self.counters["requests_429"] / total,
                "fallback_rate": self.counters["fallback_triggered"] / total,
                "cache_hit_rate": self.counters["cache_hit"] / max(self.counters["cache_hit"] + self.counters["cache_miss"], 1),
                "tokens_total": self._tokens_total,
            }


_metrics: MetricsCollector | None = None


def get_metrics() -> MetricsCollector:
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics
