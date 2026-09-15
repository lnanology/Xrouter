"""Test doubles shared across the suite. FakeProvider implements the same
Provider interface every real adapter implements, so router/fallback/
circuit-breaker tests never need real network access or Ollama installed."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from app.contracts.model import ModelInfo
from app.contracts.provider import ProviderCapability, ProviderConfig, ProviderHealth, ProviderStatus
from app.contracts.request import ChatCompletionRequest
from app.contracts.response import ChatCompletionChoice, ChatCompletionChunk, ChatCompletionChunkChoice, ChatCompletionResponse, Usage
from app.core.errors import ProviderError
from app.providers.base import Provider


class FakeProvider(Provider):
    """Scripted provider: `behavior` controls what chat()/stream_chat() do.

    behavior: "success" | "timeout" | "rate_limit" | "server_error" | "auth_error" | callable
    """

    def __init__(
        self, config: ProviderConfig, models: list[ModelInfo], behavior="success",
        fail_after_chunks: int | None = None, delay_seconds: float = 0.0,
        content: str = "ok", finish_reason: str = "stop", tool_calls: list[dict] | None = None,
        responses: list[dict] | None = None,
    ):
        super().__init__(config)
        self._models = models
        self.behavior = behavior
        self.fail_after_chunks = fail_after_chunks
        self.delay_seconds = delay_seconds
        self.content = content
        self.finish_reason = finish_reason
        self.tool_calls = tool_calls
        # Optional script: one dict of {content?, finish_reason?, tool_calls?}
        # per call, consumed in order (the last entry repeats once
        # exhausted). Lets a single FakeProvider stand in for a real model
        # across a whole planner retry loop -- e.g. an invalid submit_plan
        # call first, then a valid one -- without needing multiple fake
        # providers wired into the same test.
        self.responses = responses
        self.call_count = 0
        self.cancelled = False
        self.last_request: ChatCompletionRequest | None = None
        # Every request this provider ever received, in order -- lets a
        # test inspect an *intermediate* call (e.g. the second of several
        # sequential calls in a retry/replan loop), which last_request
        # alone can't do once later calls have overwritten it.
        self.request_log: list[ChatCompletionRequest] = []

    def capabilities(self) -> set[ProviderCapability]:
        return {ProviderCapability.CHAT, ProviderCapability.STREAMING}

    async def list_models(self) -> list[ModelInfo]:
        return self._models

    async def health(self) -> ProviderHealth:
        return ProviderHealth(status=ProviderStatus.HEALTHY, latency_ms=10.0, success_rate=1.0)

    def _maybe_raise(self):
        self.call_count += 1
        if callable(self.behavior):
            self.behavior(self.call_count)
            return
        if self.behavior == "timeout":
            from app.core.errors import ProviderTimeoutError

            raise ProviderTimeoutError("simulated timeout", provider_id=self.id)
        if self.behavior == "rate_limit":
            from app.core.errors import ProviderRateLimitError

            raise ProviderRateLimitError("simulated 429", provider_id=self.id, retry_after=0.01)
        if self.behavior == "server_error":
            from app.core.errors import ProviderServerError

            raise ProviderServerError("simulated 500", provider_id=self.id)
        if self.behavior == "auth_error":
            from app.core.errors import ProviderAuthError

            raise ProviderAuthError("simulated bad key", provider_id=self.id)

    async def chat(self, model: str, request: ChatCompletionRequest) -> ChatCompletionResponse:
        self.last_request = request
        self.request_log.append(request)
        if self.delay_seconds:
            try:
                await asyncio.sleep(self.delay_seconds)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        self._maybe_raise()
        step: dict = {}
        if self.responses:
            step = self.responses[min(self.call_count - 1, len(self.responses) - 1)]
        content = step.get("content", self.content)
        finish_reason = step.get("finish_reason", self.finish_reason)
        tool_calls = step.get("tool_calls", self.tool_calls)
        message: dict = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        return ChatCompletionResponse(
            model=f"{self.id}/{model}",
            choices=[ChatCompletionChoice(index=0, message=message, finish_reason=finish_reason)],
            usage=Usage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
        )

    async def stream_chat(self, model: str, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionChunk]:
        self._maybe_raise()
        n_chunks = 3
        for i in range(n_chunks):
            if self.fail_after_chunks is not None and i >= self.fail_after_chunks:
                raise ProviderError("simulated mid-stream failure", provider_id=self.id)
            yield ChatCompletionChunk(
                id="chatcmpl-test",
                model=f"{self.id}/{model}",
                choices=[ChatCompletionChunkChoice(index=0, delta={"content": f"chunk{i}"}, finish_reason=None)],
            )


async def build_test_engine(tmp_path, provider_specs: dict[str, dict], **routing_overrides):
    """Assembles a real ChatEngine wired to FakeProviders instead of real
    adapters — the same pieces app.core.lifecycle.startup() assembles into
    an AppContext, minus config-driven provider construction. Lets tests
    that need ChatEngine.handle_chat() end-to-end (the DAG executor, e.g.)
    run without a real Ollama/cloud provider available.

    provider_specs: {provider_id: {kwarg: value, ...}} passed straight
    through to FakeProvider (behavior, content, delay_seconds, ...).
    routing_overrides: passed straight through to RoutingConfig; defaults
    task_aware_policy to False so tests get a predictable routing policy
    unless they override it."""
    from app.cache.manager import CacheManager
    from app.core.config import CacheConfig, RoutingConfig, ServerConfig, Settings
    from app.core.context import AppContext
    from app.core.engine import ChatEngine
    from app.core.registry import ModelRegistry, ProviderRegistry
    from app.observability.events import EventBus
    from app.observability.metrics import MetricsCollector
    from app.quota.tracker import QuotaTracker
    from app.reliability.circuit_breaker import CircuitBreakerRegistry
    from app.routing.router import AdaptiveRouter
    from app.routing.scheduler import ConcurrencyLimiter
    from app.storage.database import Database
    from app.storage.repositories.metrics import MetricsRepository
    from app.storage.repositories.model import ModelRepository
    from app.storage.repositories.provider import ProviderRepository
    from app.storage.repositories.request import RequestRepository

    db_path = str(tmp_path / "test.sqlite3")
    db = Database(db_path)
    await db.init()

    providers = ProviderRegistry()
    limiter = ConcurrencyLimiter(global_limit=32)
    for pid, spec in provider_specs.items():
        cfg = ProviderConfig(id=pid, name=pid, type="openai_compatible", timeout_seconds=2.0)
        fake = FakeProvider(cfg, [make_model(pid)], **spec)
        providers.register(fake, cfg)
        limiter.configure_provider(pid, 4)

    models = ModelRegistry()
    await models.refresh(providers)

    circuits = CircuitBreakerRegistry()
    quota = QuotaTracker()
    events = EventBus()
    metrics = MetricsCollector()

    routing_defaults = {"task_aware_policy": False}
    routing_defaults.update(routing_overrides)
    routing = RoutingConfig(**routing_defaults)
    settings = Settings(server=ServerConfig(), routing=routing, cache=CacheConfig(enabled=False), providers={}, raw_routing={})

    router = AdaptiveRouter(providers, models, circuits, quota, default_policy=routing.default_policy)
    cache = CacheManager(settings.cache, db_path)

    ctx = AppContext(
        settings=settings, providers=providers, models=models, circuits=circuits, quota=quota,
        router=router, limiter=limiter, cache=cache, metrics=metrics, events=events, db=db,
        provider_repo=ProviderRepository(db), model_repo=ModelRepository(db), request_repo=RequestRepository(db),
        metrics_repo=MetricsRepository(db), health_monitor=None, performance=None,
    )
    return ChatEngine(ctx)


def make_model(provider_id: str, name: str = "test-model", **overrides) -> ModelInfo:
    defaults = dict(
        id=name, provider_id=provider_id, name=name, capabilities=["chat", "streaming", "tools"],
        supports_streaming=True, supports_tools=True, quality_score=0.7, speed_score=0.7, reliability_score=0.9, cost_score=0.8,
    )
    defaults.update(overrides)
    return ModelInfo(**defaults)
