"""Lightweight per-request trace span, used to log the pipeline stages a
request passed through (selected -> attempts -> completed) without pulling
in an external tracing dependency in Phase 1."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Span:
    request_id: str
    stages: list[dict[str, Any]] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    def mark(self, stage: str, **data: Any) -> None:
        self.stages.append({"stage": stage, "t": time.time() - self.started_at, **data})

    def duration_ms(self) -> float:
        return (time.time() - self.started_at) * 1000
