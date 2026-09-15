"""Registry of tools XRouter can execute on a node's behalf, built once
at startup from config/tools.yaml (app/tools/factory.py) the same way
app/core/registry.py's ProviderRegistry is built from providers.yaml. A
node opts into a registered tool by name (DagNodeRequest.enable_tools /
PlanNodeSpec.enable_tools); app/execution/tool_loop.py is the only thing
that ever calls .execute() on what this registry hands back."""
from __future__ import annotations

from app.tools.base import ExecutableTool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ExecutableTool] = {}

    def register(self, tool: ExecutableTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> ExecutableTool | None:
        """None both when the name was never registered and when it was
        registered but isn't configured -- callers never need to check
        .configured themselves, an unconfigured tool is simply "not
        there", same as a disabled provider is absent from routing
        candidates."""
        tool = self._tools.get(name)
        return tool if tool is not None and tool.configured else None

    def available_names(self) -> list[str]:
        return [name for name, tool in self._tools.items() if tool.configured]

    async def close_all(self) -> None:
        for tool in self._tools.values():
            try:
                await tool.close()
            except Exception:  # noqa: BLE001 -- one tool's shutdown must never block another's
                pass
