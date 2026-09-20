"""Plugin manifest contract, consumed by app/plugins/loader.py."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PluginManifest:
    name: str
    version: str
    module: str = ""
    capabilities: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    enabled: bool = False


@dataclass
class PluginLoadReport:
    """Returned by load_plugins(); attached to AppContext.plugins and
    surfaced via GET /admin/plugins. `failed` maps a plugin name to the
    reason it was skipped (import error, missing dependency, etc.) --
    never raised, matching every other registry's fail-open contract."""
    loaded: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)

    def snapshot(self) -> dict:
        return {"loaded": list(self.loaded), "failed": dict(self.failed)}
