"""Settings.load()'s config/plugins.yaml parsing -- mirrors the existing
providers.yaml/tools.yaml parsing loops exactly (app/core/config.py).
No dedicated config test file existed before the plugin loader; this is
the first, scoped to the plugins field only."""
from __future__ import annotations

from app.core.config import Settings


def test_load_with_no_plugins_yaml_present_defaults_to_empty(tmp_path):
    settings = Settings.load(config_dir=tmp_path)  # no plugins.yaml at all
    assert settings.plugins == {}


def test_load_parses_a_full_plugin_entry(tmp_path):
    (tmp_path / "plugins.yaml").write_text(
        """
plugins:
  my_plugin:
    version: "1.2.3"
    module: "my_package.my_plugin"
    capabilities: ["provider:my_type"]
    dependencies: ["other_plugin"]
    config:
      some_key: "some_value"
    enabled: true
"""
    )
    settings = Settings.load(config_dir=tmp_path)

    assert set(settings.plugins.keys()) == {"my_plugin"}
    manifest = settings.plugins["my_plugin"]
    assert manifest.name == "my_plugin"
    assert manifest.version == "1.2.3"
    assert manifest.module == "my_package.my_plugin"
    assert manifest.capabilities == ["provider:my_type"]
    assert manifest.dependencies == ["other_plugin"]
    assert manifest.config == {"some_key": "some_value"}
    assert manifest.enabled is True


def test_load_defaults_enabled_to_false_when_omitted(tmp_path):
    (tmp_path / "plugins.yaml").write_text(
        """
plugins:
  bare_plugin:
    module: "some.module"
"""
    )
    settings = Settings.load(config_dir=tmp_path)
    assert settings.plugins["bare_plugin"].enabled is False
