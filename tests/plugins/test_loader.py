"""load_plugins() is exercised against real, genuinely-importable Python
fixture modules (tests/fixtures/example_plugin.py, broken_plugin.py) --
no dynamic-import mocking -- the same "real test double, not a stub"
approach FakeProvider/FakeTool already use elsewhere (tests/helpers.py)."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from app.contracts.plugin import PluginManifest
from app.plugins.loader import load_plugins
from app.providers import factory as provider_factory
from app.tools import factory as tool_factory


def _settings(plugins: dict[str, PluginManifest]):
    return SimpleNamespace(plugins=plugins)


@pytest.fixture(autouse=True)
def _isolate_registries():
    """Every plugin test mutates the same module-level _ADAPTERS/_BUILDERS
    dicts every other provider/tool test relies on, and importing a
    fixture module caches it in sys.modules so a later test's
    load_plugins() call would silently no-op on re-import. Save/restore
    both around every test, mirroring tests/tools/test_factory.py's own
    test_build_registry_one_bad_tool_does_not_break_the_others pattern."""
    original_adapters = dict(provider_factory._ADAPTERS)
    original_builders = dict(tool_factory._BUILDERS)
    for mod in ("tests.fixtures.example_plugin", "tests.fixtures.broken_plugin"):
        sys.modules.pop(mod, None)
    try:
        yield
    finally:
        provider_factory._ADAPTERS.clear()
        provider_factory._ADAPTERS.update(original_adapters)
        tool_factory._BUILDERS.clear()
        tool_factory._BUILDERS.update(original_builders)
        for mod in ("tests.fixtures.example_plugin", "tests.fixtures.broken_plugin"):
            sys.modules.pop(mod, None)


def test_enabled_plugin_registers_its_provider_and_tool_types():
    manifest = PluginManifest(
        name="example_plugin", version="1.0.0", module="tests.fixtures.example_plugin",
        config={"key": "value"}, enabled=True,
    )
    report = load_plugins(_settings({"example_plugin": manifest}))

    assert report.loaded == ["example_plugin"]
    assert report.failed == {}
    assert "example_plugin_provider" in provider_factory._ADAPTERS
    assert "example_plugin_tool" in tool_factory._BUILDERS

    # the plugin's own top-level code read its config back during import
    module = sys.modules["tests.fixtures.example_plugin"]
    assert module.loaded_config == {"key": "value"}


def test_disabled_plugin_is_skipped_and_never_imported():
    manifest = PluginManifest(
        name="example_plugin", version="1.0.0", module="tests.fixtures.example_plugin", enabled=False,
    )
    report = load_plugins(_settings({"example_plugin": manifest}))

    assert report.loaded == []
    assert report.failed == {}
    assert "tests.fixtures.example_plugin" not in sys.modules
    assert "example_plugin_provider" not in provider_factory._ADAPTERS


def test_plugin_with_an_unmet_dependency_is_recorded_as_failed_not_raised():
    manifest = PluginManifest(
        name="example_plugin", version="1.0.0", module="tests.fixtures.example_plugin",
        dependencies=["never_declared"], enabled=True,
    )
    report = load_plugins(_settings({"example_plugin": manifest}))  # must not raise

    assert report.loaded == []
    assert "missing dependency" in report.failed["example_plugin"]
    assert "tests.fixtures.example_plugin" not in sys.modules


def test_dependency_declared_earlier_in_the_dict_loads_successfully():
    # declaration order IS the required load order -- no topological sort
    dep = PluginManifest(name="dep", version="1.0.0", module="tests.fixtures.example_plugin", enabled=True)
    dependent = PluginManifest(
        name="dependent", version="1.0.0", module="tests.fixtures.example_plugin",
        dependencies=["dep"], enabled=True,
    )
    report = load_plugins(_settings({"dep": dep, "dependent": dependent}))

    assert report.loaded == ["dep", "dependent"]
    assert report.failed == {}


def test_import_error_inside_a_plugin_is_caught_and_recorded_as_failed():
    manifest = PluginManifest(
        name="broken", version="1.0.0", module="tests.fixtures.broken_plugin", enabled=True,
    )
    report = load_plugins(_settings({"broken": manifest}))  # must not raise

    assert report.loaded == []
    assert "boom during plugin import" in report.failed["broken"]


def test_one_bad_plugin_does_not_block_a_later_good_one():
    broken = PluginManifest(name="broken", version="1.0.0", module="tests.fixtures.broken_plugin", enabled=True)
    good = PluginManifest(name="example_plugin", version="1.0.0", module="tests.fixtures.example_plugin", enabled=True)
    report = load_plugins(_settings({"broken": broken, "example_plugin": good}))

    assert report.loaded == ["example_plugin"]
    assert "broken" in report.failed


def test_plugin_with_no_module_path_is_recorded_as_failed():
    manifest = PluginManifest(name="example_plugin", version="1.0.0", enabled=True)  # module="" default
    report = load_plugins(_settings({"example_plugin": manifest}))

    assert report.loaded == []
    assert "no module path" in report.failed["example_plugin"]


def test_no_plugins_configured_returns_an_empty_report():
    report = load_plugins(_settings({}))
    assert report.loaded == []
    assert report.failed == {}
