from types import SimpleNamespace

from app.core.config import ToolConfig
from app.tools.factory import build_tool_registry, register_builder
from app.tools.web_fetch import WebFetchTool
from app.tools.web_search import WebSearchTool


def _settings(tools: dict[str, ToolConfig]):
    return SimpleNamespace(tools=tools)


def test_build_registry_empty_when_no_tools_configured():
    registry = build_tool_registry(_settings({}))
    assert registry.available_names() == []


def test_build_registry_skips_disabled_tools(monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "sk-1")
    settings = _settings({
        "web_search": ToolConfig(id="web_search", enabled=False, api_key_env="FAKE_KEY"),
    })
    registry = build_tool_registry(settings)
    assert registry.available_names() == []


def test_build_registry_builds_a_configured_web_search_tool(monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "sk-1")
    settings = _settings({
        "web_search": ToolConfig(id="web_search", enabled=True, api_key_env="FAKE_KEY", base_url="https://x.example.com"),
    })
    registry = build_tool_registry(settings)
    assert registry.available_names() == ["web_search"]
    tool = registry.get("web_search")
    assert isinstance(tool, WebSearchTool)


def test_build_registry_tool_stays_unconfigured_without_api_key(monkeypatch):
    monkeypatch.delenv("MISSING_KEY", raising=False)
    settings = _settings({
        "web_search": ToolConfig(id="web_search", enabled=True, api_key_env="MISSING_KEY"),
    })
    registry = build_tool_registry(settings)
    # enabled in config but no API key present -> gracefully absent, not a crash
    assert registry.available_names() == []
    assert registry.get("web_search") is None


def test_build_registry_builds_a_configured_web_fetch_tool(monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "sk-1")
    settings = _settings({
        "web_fetch": ToolConfig(id="web_fetch", enabled=True, api_key_env="FAKE_KEY", base_url="https://x.example.com"),
    })
    registry = build_tool_registry(settings)
    assert registry.available_names() == ["web_fetch"]
    tool = registry.get("web_fetch")
    assert isinstance(tool, WebFetchTool)


def test_build_registry_skips_unknown_tool_id_without_crashing(monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "sk-1")
    settings = _settings({
        "not_a_real_tool": ToolConfig(id="not_a_real_tool", enabled=True, api_key_env="FAKE_KEY"),
    })
    registry = build_tool_registry(settings)  # must not raise
    assert registry.available_names() == []


def test_build_registry_one_bad_tool_does_not_break_the_others(monkeypatch):
    monkeypatch.setenv("GOOD_KEY", "sk-1")

    class ExplodingTool:
        def __init__(self, *a, **k):
            raise RuntimeError("boom during init")

    settings = _settings({
        "web_search": ToolConfig(id="web_search", enabled=True, api_key_env="GOOD_KEY"),
        "broken": ToolConfig(id="broken", enabled=True, api_key_env="GOOD_KEY"),
    })
    # monkeypatch the builder table just for "broken" so it explodes
    from app.tools import factory as factory_module

    original_builders = dict(factory_module._BUILDERS)
    factory_module._BUILDERS["broken"] = lambda cfg: ExplodingTool()
    try:
        registry = build_tool_registry(settings)  # must not raise
    finally:
        factory_module._BUILDERS.clear()
        factory_module._BUILDERS.update(original_builders)

    assert registry.available_names() == ["web_search"]


def test_register_builder_adds_a_new_tool_type_usable_by_build_tool_registry():
    # the plugin-loader-facing hook (app/plugins/loader.py calls this
    # symmetric to app.providers.factory.register_adapter)
    from app.tools import factory as factory_module

    class PluginTool:
        name = "plugin_tool"
        schema: dict = {"type": "function", "function": {"name": "plugin_tool"}}

        def __init__(self, cfg):
            self.cfg = cfg

        @property
        def configured(self) -> bool:
            return True

        async def execute(self, arguments: dict) -> str:
            return "ok"

        async def close(self) -> None:
            pass

    original_builders = dict(factory_module._BUILDERS)
    try:
        register_builder("plugin_tool", lambda cfg: PluginTool(cfg))
        settings = _settings({"plugin_tool": ToolConfig(id="plugin_tool", enabled=True)})
        registry = build_tool_registry(settings)
        assert registry.available_names() == ["plugin_tool"]
        assert isinstance(registry.get("plugin_tool"), PluginTool)
    finally:
        factory_module._BUILDERS.clear()
        factory_module._BUILDERS.update(original_builders)
