"""Base interface every XRouter-executable tool implements: a name, an
OpenAI-format tool schema, whether it's actually configured/usable, and
an async execute() that performs the real action and returns plain text
to feed back to the model. Swappable/config-driven like every provider
adapter (see app/providers/base.py): app/tools/factory.py wires concrete
tools up from config/tools.yaml, app/tools/registry.py holds them, and
app/execution/tool_loop.py is the only thing that ever calls execute() --
a DAG/Planner node never talks to a tool directly."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ExecutableTool(Protocol):
    name: str
    schema: dict[str, Any]

    @property
    def configured(self) -> bool:
        """False disables the tool gracefully -- same rule a provider with
        a missing API key follows: never crash, just stay unavailable."""
        ...

    async def execute(self, arguments: dict[str, Any]) -> str: ...

    async def close(self) -> None: ...
