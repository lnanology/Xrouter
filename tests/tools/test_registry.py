import pytest

from app.tools.registry import ToolRegistry
from tests.helpers import FakeTool


def test_get_returns_none_for_never_registered_name():
    registry = ToolRegistry()
    assert registry.get("ghost") is None


def test_get_returns_the_tool_when_registered_and_configured():
    registry = ToolRegistry()
    tool = FakeTool(name="web_search", configured=True)
    registry.register(tool)
    assert registry.get("web_search") is tool


def test_get_returns_none_when_registered_but_not_configured():
    registry = ToolRegistry()
    registry.register(FakeTool(name="web_search", configured=False))
    assert registry.get("web_search") is None


def test_available_names_only_lists_configured_tools():
    registry = ToolRegistry()
    registry.register(FakeTool(name="configured_one", configured=True))
    registry.register(FakeTool(name="not_configured", configured=False))
    assert registry.available_names() == ["configured_one"]


def test_available_names_empty_for_empty_registry():
    assert ToolRegistry().available_names() == []


@pytest.mark.asyncio
async def test_close_all_closes_every_registered_tool():
    registry = ToolRegistry()
    a = FakeTool(name="a")
    b = FakeTool(name="b")
    registry.register(a)
    registry.register(b)
    await registry.close_all()
    assert a.closed is True
    assert b.closed is True


@pytest.mark.asyncio
async def test_close_all_survives_one_tool_raising():
    class BrokenTool(FakeTool):
        async def close(self):
            raise RuntimeError("boom")

    registry = ToolRegistry()
    registry.register(BrokenTool(name="broken"))
    good = FakeTool(name="good")
    registry.register(good)
    await registry.close_all()  # must not raise
    assert good.closed is True
