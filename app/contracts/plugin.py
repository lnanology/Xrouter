"""Plugin manifest contract. Reserved for Phase 2+; not loaded in Phase 1."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PluginManifest:
    name: str
    version: str
    capabilities: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    enabled: bool = False
