"""Config-driven settings. All provider/model/routing config comes from
config/*.yaml; secrets come from environment variables (.env). Nothing is
hardcoded in code."""
from __future__ import annotations

import logging
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from app.contracts.provider import ProviderConfig

CONFIG_DIR = Path(os.environ.get("XROUTER_CONFIG_DIR", Path(__file__).resolve().parents[2] / "config"))
DATA_DIR = Path(os.environ.get("XROUTER_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 20128
    admin_token: str | None = None
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    request_timeout_seconds: float = 60.0
    max_request_body_bytes: int = 5_000_000
    global_concurrency_limit: int = 64
    per_client_rate_limit_per_minute: int = 120


@dataclass
class RoutingConfig:
    default_policy: str = "balanced"
    max_fallback_attempts: int = 3
    race_mode_enabled: bool = False
    # How many top-ranked candidates a raced request dispatches concurrently
    # (only relevant when race_mode_enabled is True and the client requests
    # race=true). Clamped at use to [1, number of available candidates].
    race_candidate_count: int = 2
    # Confidence/Quality-Gate Engine (Phase 2): off by default, same reason
    # as race mode — a failed gate triggers an extra provider call, so it
    # shouldn't turn on silently. When enabled, a response scoring below
    # quality_gate_min_score gets retried with the next untried candidate,
    # up to max_quality_retries times (non-streaming requests only).
    quality_gate_enabled: bool = False
    quality_gate_min_score: float = 0.5
    max_quality_retries: int = 1
    # DAG Executor (Phase 2): caps how many nodes a single client-submitted
    # DAG may contain, so one request can't fan out into an unbounded
    # number of provider calls.
    max_dag_nodes: int = 20
    # Planner (Phase 3 groundwork): the routing policy used for the one
    # planning call itself (not the resulting DAG nodes, which each pick
    # their own policy the normal way) -- "quality" by default, since a
    # bad plan wastes every node run under it. How many times the planner
    # re-prompts (feeding back the specific parse/validation error) before
    # giving up on a request.
    planner_routing_policy: str = "quality"
    max_plan_retries: int = 2
    # Verifier (Phase 3 groundwork): bounds how many times a failed
    # verification (PlanRequest.verify=true) can trigger a fresh
    # plan+execute cycle. Only consulted when the request opts in.
    max_verify_retries: int = 1
    # Tool-execution loop (Phase 3): how many call -> tool -> call
    # round-trips a single node's XRouter-executed tool use may take
    # (DagNodeRequest.enable_tools) before giving up and returning
    # whatever the model last said, rather than looping forever.
    max_tool_iterations: int = 3
    # Critic (Phase 3): how many times a single DAG node re-runs itself
    # with the Critic's feedback folded in (DagNodeRequest.critique)
    # before accepting whatever the last attempt produced, rather than
    # retrying forever.
    max_critique_retries: int = 1
    # Phase 2: when true, an unspecified client `routing_policy` is chosen
    # by the Task Classifier (task type + complexity) instead of always
    # falling back to `default_policy`. `default_policy` is still used when
    # this is false, and still wins for any task type the classifier can't
    # place confidently.
    task_aware_policy: bool = True


@dataclass
class ToolConfig:
    id: str
    enabled: bool = False
    api_key_env: str | None = None
    base_url: str | None = None
    timeout_seconds: float = 10.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheConfig:
    enabled: bool = True
    l1_max_entries: int = 512
    default_ttl_seconds: float = 300.0
    volatile_keywords: list[str] = field(
        default_factory=lambda: ["today", "now", "current", "latest", "news", "price", "stock", "weather"]
    )


@dataclass
class BenchmarkConfig:
    """Automated Benchmark (Phase 5's first piece, section 三十六): off by
    default. Unlike the always-on HealthMonitor (its health() polling is
    assumed cheap and provider-defined), this fires a real chat completion
    against every enabled provider on a timer with no per-request trigger --
    same "costs real money, must not turn on silently" reasoning already
    established for race_mode_enabled / quality_gate_enabled /
    trace_evidence / trace_counterfactual. See app/reliability/benchmark.py."""
    enabled: bool = False
    interval_seconds: float = 3600.0
    prompt: str = "Reply with the single word: OK"
    max_tokens: int = 8


@dataclass
class ABRoutingConfig:
    """A/B Routing (Phase 5's second piece, section 三十六): off by default.
    Unlike Automated Benchmark (a timer that costs real provider calls),
    turning this on changes how *every* non-pinned request already being
    served gets routed -- a real behavior change for live traffic, not an
    extra background call -- so it gets the same "must not turn on
    silently" default as everything else in this family. See
    app/routing/ab_router.py."""
    enabled: bool = False
    variants: list[str] = field(default_factory=lambda: ["quality", "balanced"])


@dataclass
class PolicyLearningConfig:
    """Policy Learning (Phase 5's third piece, section 三十六): off by
    default. Like A/B Routing, turning this on changes how live traffic
    gets scored -- app/routing/router.py's resolve_policy starts returning
    nudged weights instead of the static defaults -- so it gets the same
    "must not turn on silently" default as everything else in this family.
    Depends on A/B Routing already being enabled and trafficked: with no
    ab_results data, PolicyLearner.run_once() just reports
    insufficient_samples every time -- an expected consequence of the
    dependency, not a bug. See app/routing/policy_learner.py."""
    enabled: bool = False
    min_samples: int = 20
    learning_rate: float = 0.15
    min_margin: float = 0.05
    latency_weight_per_second: float = 0.05
    quality_weight: float = 0.1


@dataclass
class EvolutionConfig:
    """Evolution Engine (Phase 5's fourth piece, section 三十六): off by
    default. Unlike A/B Routing/Policy Learning (config flags that change
    per-request behavior), this runs its own background timer -- same
    "must not turn on silently" reasoning as Automated Benchmark. Each
    cycle: evaluate any pending nudge from a prior cycle against fresh
    post-nudge evidence (rolling it back if real-world performance got
    worse), then ask PolicyLearner for a new nudge. Depends on Policy
    Learning already being enabled: if policy_learning.enabled is False,
    PolicyLearner.weights_for() never applies anything this computes, so
    cycles just run against routing that isn't actually using the result
    -- an expected consequence of the dependency chain, not a bug. See
    app/routing/evolution_engine.py."""
    enabled: bool = False
    interval_seconds: float = 3600.0
    evaluation_samples: int = 20
    rollback_tolerance: float = 0.02


@dataclass
class SelfHealingConfig:
    """Self-healing (Phase 5's fifth and last piece): off by default --
    same "must not turn on silently" reasoning as every other opt-in
    Phase 5 piece, since this is the one background task that actually
    takes ProviderRegistry.set_enabled() out of human hands. Each cycle
    correlates two signals that already exist for free (ProviderRegistry.
    health_of() from HealthMonitor's own polling, and
    CircuitBreakerRegistry's per-provider state from real request
    traffic) -- it never re-probes a provider itself. A provider that's
    looked bad on either signal for confirm_cycles consecutive checks
    gets auto-disabled; one that's looked good for confirm_cycles
    consecutive checks while self-disabled gets auto re-enabled. Never
    touches a provider an admin disabled directly -- see
    SelfHealer.forget() in app/reliability/self_healer.py."""
    enabled: bool = False
    interval_seconds: float = 30.0
    confirm_cycles: int = 3


@dataclass
class Settings:
    server: ServerConfig
    routing: RoutingConfig
    cache: CacheConfig
    benchmark: BenchmarkConfig
    ab_routing: ABRoutingConfig
    policy_learning: PolicyLearningConfig
    evolution: EvolutionConfig
    self_healing: SelfHealingConfig
    providers: dict[str, ProviderConfig]
    raw_routing: dict[str, Any]
    model_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    tools: dict[str, ToolConfig] = field(default_factory=dict)

    @classmethod
    def load(cls, config_dir: Path | None = None) -> "Settings":
        load_dotenv()  # loads .env into os.environ if present; no-op otherwise
        cdir = config_dir or CONFIG_DIR

        raw_config = _load_yaml(cdir / "config.yaml")
        raw_providers = _load_yaml(cdir / "providers.yaml")
        raw_routing = _load_yaml(cdir / "routing.yaml")
        raw_models = _load_yaml(cdir / "models.yaml")
        model_overrides = raw_models.get("models", {}) or {}
        raw_tools = _load_yaml(cdir / "tools.yaml")

        server_raw = raw_config.get("server", {})
        server = ServerConfig(
            host=server_raw.get("host", ServerConfig.host),
            port=int(server_raw.get("port", ServerConfig.port)),
            admin_token=server_raw.get("admin_token") or os.environ.get("XROUTER_ADMIN_TOKEN"),
            cors_origins=server_raw.get("cors_origins", ["*"]),
            request_timeout_seconds=float(server_raw.get("request_timeout_seconds", 60.0)),
            max_request_body_bytes=int(server_raw.get("max_request_body_bytes", 5_000_000)),
            global_concurrency_limit=int(server_raw.get("global_concurrency_limit", 64)),
            per_client_rate_limit_per_minute=int(server_raw.get("per_client_rate_limit_per_minute", 120)),
        )

        routing_raw = raw_routing.get("routing", {})
        routing = RoutingConfig(
            default_policy=routing_raw.get("default_policy", "balanced"),
            max_fallback_attempts=int(routing_raw.get("max_fallback_attempts", 3)),
            race_mode_enabled=bool(routing_raw.get("race_mode_enabled", False)),
            race_candidate_count=int(routing_raw.get("race_candidate_count", 2)),
            quality_gate_enabled=bool(routing_raw.get("quality_gate_enabled", False)),
            quality_gate_min_score=float(routing_raw.get("quality_gate_min_score", 0.5)),
            max_quality_retries=int(routing_raw.get("max_quality_retries", 1)),
            max_dag_nodes=int(routing_raw.get("max_dag_nodes", 20)),
            planner_routing_policy=routing_raw.get("planner_routing_policy", "quality"),
            max_plan_retries=int(routing_raw.get("max_plan_retries", 2)),
            max_verify_retries=int(routing_raw.get("max_verify_retries", 1)),
            max_tool_iterations=int(routing_raw.get("max_tool_iterations", 3)),
            max_critique_retries=int(routing_raw.get("max_critique_retries", 1)),
            task_aware_policy=bool(routing_raw.get("task_aware_policy", True)),
        )

        cache_raw = raw_config.get("cache", {})
        cache = CacheConfig(
            enabled=bool(cache_raw.get("enabled", True)),
            l1_max_entries=int(cache_raw.get("l1_max_entries", 512)),
            default_ttl_seconds=float(cache_raw.get("default_ttl_seconds", 300.0)),
            volatile_keywords=cache_raw.get("volatile_keywords", CacheConfig().volatile_keywords),
        )

        benchmark_raw = raw_config.get("benchmark", {})
        benchmark = BenchmarkConfig(
            enabled=bool(benchmark_raw.get("enabled", False)),
            interval_seconds=float(benchmark_raw.get("interval_seconds", 3600.0)),
            prompt=benchmark_raw.get("prompt", BenchmarkConfig.prompt),
            max_tokens=int(benchmark_raw.get("max_tokens", 8)),
        )

        ab_routing_raw = raw_config.get("ab_routing", {})
        ab_routing = ABRoutingConfig(
            enabled=bool(ab_routing_raw.get("enabled", False)),
            variants=ab_routing_raw.get("variants", ABRoutingConfig().variants),
        )

        policy_learning_raw = raw_config.get("policy_learning", {})
        policy_learning = PolicyLearningConfig(
            enabled=bool(policy_learning_raw.get("enabled", False)),
            min_samples=int(policy_learning_raw.get("min_samples", 20)),
            learning_rate=float(policy_learning_raw.get("learning_rate", 0.15)),
            min_margin=float(policy_learning_raw.get("min_margin", 0.05)),
            latency_weight_per_second=float(policy_learning_raw.get("latency_weight_per_second", 0.05)),
            quality_weight=float(policy_learning_raw.get("quality_weight", 0.1)),
        )

        evolution_raw = raw_config.get("evolution", {})
        evolution = EvolutionConfig(
            enabled=bool(evolution_raw.get("enabled", False)),
            interval_seconds=float(evolution_raw.get("interval_seconds", 3600.0)),
            evaluation_samples=int(evolution_raw.get("evaluation_samples", 20)),
            rollback_tolerance=float(evolution_raw.get("rollback_tolerance", 0.02)),
        )

        self_healing_raw = raw_config.get("self_healing", {})
        self_healing = SelfHealingConfig(
            enabled=bool(self_healing_raw.get("enabled", False)),
            interval_seconds=float(self_healing_raw.get("interval_seconds", 30.0)),
            confirm_cycles=int(self_healing_raw.get("confirm_cycles", 3)),
        )

        providers: dict[str, ProviderConfig] = {}
        for pid, pcfg in (raw_providers.get("providers") or {}).items():
            providers[pid] = ProviderConfig(
                id=pid,
                name=pcfg.get("name", pid),
                type=pcfg.get("type", pid),
                enabled=bool(pcfg.get("enabled", False)),
                base_url=pcfg.get("base_url"),
                api_key_env=pcfg.get("api_key_env"),
                timeout_seconds=float(pcfg.get("timeout_seconds", 30.0)),
                max_concurrency=int(pcfg.get("max_concurrency", 4)),
                priority=int(pcfg.get("priority", 0)),
                extra=pcfg.get("extra", {}) or {},
            )

        tools: dict[str, ToolConfig] = {}
        for tid, tcfg in (raw_tools.get("tools") or {}).items():
            tools[tid] = ToolConfig(
                id=tid,
                enabled=bool(tcfg.get("enabled", False)),
                api_key_env=tcfg.get("api_key_env"),
                base_url=tcfg.get("base_url"),
                timeout_seconds=float(tcfg.get("timeout_seconds", 10.0)),
                extra=tcfg.get("extra", {}) or {},
            )

        if not server.admin_token:
            # Never leave /admin/* unauthenticated by default (section 三十三).
            server.admin_token = secrets.token_urlsafe(24)
            logging.getLogger("xrouter.config").warning(
                "No XROUTER_ADMIN_TOKEN configured — generated a random admin token for this run: %s "
                "(set XROUTER_ADMIN_TOKEN in .env to persist it across restarts)",
                server.admin_token,
            )

        return cls(
            server=server, routing=routing, cache=cache, benchmark=benchmark, ab_routing=ab_routing,
            policy_learning=policy_learning, evolution=evolution, self_healing=self_healing, providers=providers,
            raw_routing=raw_routing, model_overrides=model_overrides, tools=tools,
        )


_settings: Settings | None = None


def get_settings(reload: bool = False) -> Settings:
    global _settings
    if _settings is None or reload:
        _settings = Settings.load()
    return _settings
