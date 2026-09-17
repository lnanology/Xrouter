"""Startup/shutdown wiring. Startup must never crash because one optional
provider is misconfigured (section 三十七); shutdown must be graceful for
every background task and DB connection."""
from __future__ import annotations

from app.cache.manager import CacheManager
from app.core.config import DATA_DIR, Settings, get_settings
from app.core.context import AppContext
from app.core.registry import ModelRegistry, ProviderRegistry
from app.observability.events import get_event_bus
from app.observability.logging import get_logger
from app.observability.metrics import get_metrics
from app.quota.tracker import get_quota_tracker
from app.reliability.benchmark import BenchmarkScheduler
from app.reliability.circuit_breaker import CircuitBreakerRegistry
from app.reliability.health import HealthMonitor
from app.routing.ab_router import ABRouter
from app.routing.performance_controller import PerformanceController
from app.routing.router import AdaptiveRouter
from app.routing.scheduler import ConcurrencyLimiter
from app.storage.database import Database
from app.storage.repositories.metrics import MetricsRepository
from app.storage.repositories.memory import MemoryRepository
from app.storage.repositories.model import ModelRepository
from app.storage.repositories.provider import ProviderRepository
from app.storage.repositories.request import RequestRepository
from app.tools.factory import build_tool_registry

logger = get_logger("lifecycle")


async def startup(settings: Settings | None = None) -> AppContext:
    settings = settings or get_settings()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    db_path = str(DATA_DIR / "xrouter.sqlite3")

    db = Database(db_path)
    await db.init()

    providers = ProviderRegistry.build(settings)
    models = ModelRegistry()
    await models.refresh(providers)
    models.apply_overrides(settings.model_overrides)

    circuits = CircuitBreakerRegistry()
    quota = get_quota_tracker()
    metrics = get_metrics()
    events = get_event_bus()

    limiter = ConcurrencyLimiter(global_limit=settings.server.global_concurrency_limit)
    for pid, pcfg in settings.providers.items():
        limiter.configure_provider(pid, pcfg.max_concurrency)

    provider_repo = ProviderRepository(db)
    model_repo = ModelRepository(db)
    request_repo = RequestRepository(db)
    metrics_repo = MetricsRepository(db)
    memory_repo = MemoryRepository(db)

    performance = PerformanceController(metrics_repo=metrics_repo, events=events, snapshot_interval_seconds=60.0)
    performance.start()

    router = AdaptiveRouter(
        providers, models, circuits, quota,
        default_policy=settings.routing.default_policy, performance_controller=performance,
    )
    cache = CacheManager(settings.cache, db_path)

    for pid, pcfg in settings.providers.items():
        await provider_repo.upsert(pid, pcfg.name, pcfg.type, providers.is_enabled(pid))
    await model_repo.upsert_many(models.all())

    health_monitor = HealthMonitor(providers, circuits, provider_repo, interval_seconds=20.0)
    health_monitor.start()

    benchmark_scheduler = BenchmarkScheduler(
        providers, models, metrics_repo,
        prompt=settings.benchmark.prompt, max_tokens=settings.benchmark.max_tokens,
        interval_seconds=settings.benchmark.interval_seconds,
    )
    if settings.benchmark.enabled:
        benchmark_scheduler.start()

    ab_router = ABRouter(settings.ab_routing.variants, metrics_repo, enabled=settings.ab_routing.enabled)

    tools = build_tool_registry(settings)

    logger.info(
        "startup complete: %d provider(s) configured, %d enabled, %d model(s) discovered, %d tool(s) available",
        len(settings.providers), sum(1 for p in settings.providers if providers.is_enabled(p)), len(models.all()),
        len(tools.available_names()),
    )

    return AppContext(
        settings=settings, providers=providers, models=models, circuits=circuits, quota=quota,
        router=router, limiter=limiter, cache=cache, metrics=metrics, events=events, db=db,
        provider_repo=provider_repo, model_repo=model_repo, request_repo=request_repo,
        metrics_repo=metrics_repo, health_monitor=health_monitor, performance=performance, tools=tools,
        memory_repo=memory_repo, benchmark_scheduler=benchmark_scheduler, ab_router=ab_router,
    )


async def shutdown(ctx: AppContext) -> None:
    logger.info("shutdown starting")
    await ctx.health_monitor.stop()
    await ctx.performance.stop()
    await ctx.benchmark_scheduler.stop()
    await ctx.providers.close_all()
    await ctx.tools.close_all()
    logger.info("shutdown complete")
