"""Real, importable fixture plugin used by tests/plugins/test_loader.py
to prove app.plugins.loader.load_plugins() works end to end against a
genuine plugin module -- mirroring how FakeProvider/FakeTool already
stand in for a real Ollama/Tavily integration elsewhere (tests/helpers.py).
On import, this module registers a new provider type and a new tool type
via the exact two extension points a real third-party plugin would use
(app.providers.factory.register_adapter, app.tools.factory.register_builder),
and reads back its own config through get_plugin_config() -- proving the
loader's "config available during import" contract."""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from app.contracts.model import ModelInfo
from app.contracts.provider import ProviderCapability, ProviderHealth
from app.contracts.request import ChatCompletionRequest
from app.contracts.response import ChatCompletionChunk, ChatCompletionResponse
from app.plugins.loader import get_plugin_config
from app.providers.base import Provider
from app.providers.factory import register_adapter
from app.tools.factory import register_builder

# Proves get_plugin_config() actually returns this plugin's own config
# blob (from plugins.yaml's `config:` entry) during its own import.
loaded_config: dict[str, Any] = get_plugin_config("example_plugin")


class ExamplePluginProvider(Provider):
    """Minimal but real Provider implementation -- never actually called
    in these tests, only registered, so every method is a harmless stub."""

    def capabilities(self) -> set[ProviderCapability]:
        return set()

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def health(self) -> ProviderHealth:
        return ProviderHealth()

    async def chat(self, model: str, request: ChatCompletionRequest) -> ChatCompletionResponse:
        raise NotImplementedError

    def stream_chat(self, model: str, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionChunk]:
        raise NotImplementedError


class ExamplePluginTool:
    """Minimal ExecutableTool implementation (app/tools/base.py)."""

    name = "example_plugin_tool"
    schema: dict[str, Any] = {"type": "function", "function": {"name": "example_plugin_tool"}}

    def __init__(self, cfg: Any = None):
        self.cfg = cfg

    @property
    def configured(self) -> bool:
        return True

    async def execute(self, arguments: dict[str, Any]) -> str:
        return "ok"

    async def close(self) -> None:
        pass


register_adapter("example_plugin_provider", ExamplePluginProvider)
register_builder("example_plugin_tool", lambda cfg: ExamplePluginTool(cfg))
