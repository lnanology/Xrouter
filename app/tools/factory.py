"""Builds a ToolRegistry from config/tools.yaml -- the same "one bad
entry never crashes startup" pattern app/core/registry.py's
ProviderRegistry.build() uses for providers.yaml. Adding a new tool type
means writing a class implementing app/tools/base.py's ExecutableTool
protocol and adding one entry to _BUILDERS; core/execution code never
needs to change."""
from __future__ import annotations

from app.core.config import Settings, ToolConfig
from app.observability.logging import get_logger
from app.tools.registry import ToolRegistry
from app.tools.web_fetch import WebFetchTool
from app.tools.web_search import WebSearchTool
from app.utils.secrets import read_secret

logger = get_logger("tools.factory")

_BUILDERS = {
    "web_search": lambda cfg: WebSearchTool(
        api_key=read_secret(cfg.api_key_env), base_url=cfg.base_url or "https://api.tavily.com",
        timeout_seconds=cfg.timeout_seconds,
    ),
    "web_fetch": lambda cfg: WebFetchTool(
        api_key=read_secret(cfg.api_key_env), base_url=cfg.base_url or "https://api.tavily.com",
        timeout_seconds=cfg.timeout_seconds,
    ),
}


def register_builder(type_key: str, builder) -> None:
    """Allows plugins to register new tool types without editing this
    file -- the tools-side counterpart to
    app.providers.factory.register_adapter(). Called by
    app/plugins/loader.py while importing a plugin module."""
    _BUILDERS[type_key] = builder


def build_tool_registry(settings: Settings) -> ToolRegistry:
    registry = ToolRegistry()
    for tool_id, cfg in settings.tools.items():
        if not cfg.enabled:
            continue
        builder = _BUILDERS.get(tool_id)
        if builder is None:
            logger.warning("no builder registered for tool '%s' in tools.yaml; skipping", tool_id)
            continue
        try:
            registry.register(builder(cfg))
        except Exception as e:  # a bad single tool must never crash startup
            logger.warning("failed to initialize tool '%s': %s", tool_id, e)
    return registry
