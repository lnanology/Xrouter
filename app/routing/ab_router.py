"""A/B Routing (Phase 5, section 三十六): splits real, live traffic between
two or more named routing-policy variants and records each request's real
comparative outcome, tagged by variant. Distinct from app/execution/race.py's
"race mode", which hedges within a single policy's own candidate list —
this compares different policies against each other using separate traffic.

Off by default (ABRoutingConfig.enabled=False) — see app/core/config.py for
why: turning this on is a real behavior change for every non-pinned request,
not just an extra background call."""
from __future__ import annotations

import hashlib
import uuid

from app.contracts.request import ChatCompletionRequest
from app.storage.repositories.metrics import MetricsRepository


class ABRouter:
    def __init__(self, variants: list[str], metrics_repo: MetricsRepository, enabled: bool = False):
        self._variants = list(variants)
        self._repo = metrics_repo
        self._enabled = enabled

    def assign(self, request: ChatCompletionRequest) -> str | None:
        """Returns the assigned policy variant, or None when A/B Routing is
        disabled or fewer than 2 variants are configured (a soft no-op, the
        same "gracefully do nothing" precedent used everywhere else in this
        codebase rather than raising).

        Assignment is sticky per `request.user` (hashed with md5, modulo the
        variant count) — the same user always lands on the same variant.
        XRouter has no session concept today, so a request with no `user`
        falls back to a fresh per-request uuid4: an anonymous caller gets
        per-request rather than sticky assignment, which is still
        statistically valid in aggregate, just not sticky for that caller.
        This is a documented, honest limitation, not a bug."""
        if not self._enabled or len(self._variants) < 2:
            return None
        key = request.user or str(uuid.uuid4())
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        index = int(digest, 16) % len(self._variants)
        return self._variants[index]

    async def record_outcome(
        self, variant: str, success: bool, latency_ms: float | None, quality_score: float | None = None,
    ) -> None:
        await self._repo.record_ab_result(variant, success, latency_ms, quality_score)
