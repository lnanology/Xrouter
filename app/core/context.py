"""Single dependency-injection container built once at startup and attached
to app.state. Every request handler reads shared singletons from here
instead of importing module-level globals directly, which keeps the system
testable (a test can build its own AppContext with fakes)."""
from __future__ import annotations

from dataclasses import dataclass

from app.cache.manager import CacheManager
from app.core.config import Settings
from app.core.registry import ModelRegistry, ProviderRegistry
from app.observability.events import EventBus
from app.observability.metrics import MetricsCollector
from app.quota.tracker import QuotaTracker
from app.reliability.benchmark import BenchmarkScheduler
from app.reliability.circuit_breaker import CircuitBreakerRegistry
from app.reliability.health import HealthMonitor
from app.routing.ab_router import ABRouter
from app.routing.performance_controller import PerformanceController
from app.routing.policy_learner import PolicyLearner
from app.routing.router import AdaptiveRouter
from app.routing.scheduler import ConcurrencyLimiter
from app.storage.database import Database
from app.tools.registry import ToolRegistry
from app.storage.repositories.metrics import MetricsRepository
from app.storage.repositories.memory import MemoryRepository
from app.storage.repositories.model import ModelRepository
from app.storage.repositories.provider import ProviderRepository
from app.storage.repositories.request import RequestRepository


@dataclass
class AppContext:
    settings: Settings
    providers: ProviderRegistry
    models: ModelRegistry
    circuits: CircuitBreakerRegistry
    quota: QuotaTracker
    router: AdaptiveRouter
    limiter: ConcurrencyLimiter
    cache: CacheManager
    metrics: MetricsCollector
    events: EventBus
    db: Database
    provider_repo: ProviderRepository
    model_repo: ModelRepository
    request_repo: RequestRepository
    metrics_repo: MetricsRepository
    health_monitor: HealthMonitor
    performance: PerformanceController
    tools: ToolRegistry
    memory_repo: MemoryRepository
    benchmark_scheduler: BenchmarkScheduler
    ab_router: ABRouter
    policy_learner: PolicyLearner
