"""Real, importable fixture plugin that raises at import time, used by
tests/plugins/test_loader.py to prove load_plugins() catches a bad
plugin's import error and records it in PluginLoadReport.failed rather
than crashing startup -- the same fail-open contract every other
registry in the codebase already follows."""
from __future__ import annotations

raise RuntimeError("boom during plugin import")
