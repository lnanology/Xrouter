"""Loads plugin modules declared in config/plugins.yaml. Each plugin is
a real, importable Python module whose own top-level code registers new
provider/tool *types* via app.providers.factory.register_adapter /
app.tools.factory.register_builder -- the two extension points the
codebase already exposed ahead of need. Deliberately scoped to
type-level registration only (no instance-level "register(providers,
tools)" callback, no hot-reload, no sandboxing, no dependency
topological sort, no PyPI/entry_points auto-discovery) -- rule 15, "no
unnecessary infrastructure".

Runs as the first statement in app.core.lifecycle.startup(), before
ProviderRegistry.build() and build_tool_registry() consume those
registries, so newly-registered types are visible to both.

Never crashes startup: a disabled plugin is skipped, a plugin whose
dependencies aren't loaded yet (declaration order in plugins.yaml IS the
required load order -- no topological sort) is skipped, and any
exception raised while importing one plugin is caught, logged, and
recorded as a failure -- matching every other registry's fail-open
contract (app/providers/factory.py, app/tools/factory.py)."""
from __future__ import annotations

import importlib
from typing import Any

from app.contracts.plugin import PluginLoadReport
from app.core.config import Settings
from app.observability.logging import get_logger

logger = get_logger("plugins.loader")

_plugin_config: dict[str, dict[str, Any]] = {}


def get_plugin_config(name: str) -> dict[str, Any]:
    """Callable from within a plugin module's own top-level code during
    its import, so it can read its own config blob (plugins.yaml's
    `config:` entry) without the loader needing to call back into it."""
    return _plugin_config.get(name, {})


def load_plugins(settings: Settings) -> PluginLoadReport:
    report = PluginLoadReport()
    loaded: set[str] = set()

    for name, manifest in settings.plugins.items():
        if not manifest.enabled:
            continue

        missing = [dep for dep in manifest.dependencies if dep not in loaded]
        if missing:
            report.failed[name] = f"missing dependency(ies): {', '.join(missing)}"
            logger.warning("plugin '%s' skipped: missing dependency(ies) %s", name, missing)
            continue

        if not manifest.module:
            report.failed[name] = "no module path configured"
            logger.warning("plugin '%s' skipped: no module path configured", name)
            continue

        _plugin_config[name] = manifest.config
        try:
            importlib.import_module(manifest.module)
        except Exception as e:  # a bad single plugin must never crash startup
            report.failed[name] = str(e)
            logger.warning("failed to load plugin '%s' (%s): %s", name, manifest.module, e)
            continue

        loaded.add(name)
        report.loaded.append(name)

    if report.loaded or report.failed:
        logger.info("plugins loaded: %s; failed: %s", report.loaded, list(report.failed))

    return report
