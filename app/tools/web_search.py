"""Web Search tool (Phase 3): a real, executable tool a DAG/Planner node
can invoke via `"enable_tools": ["web_search"]` -- not a fake placeholder
that just echoes the query back. Config-driven like every provider
adapter (app/providers/*): reads its API key from an env var (see
config/tools.yaml, .env.example) and disables itself gracefully if
unconfigured, the same rule a provider with no key follows. Backed by
Tavily's search API (https://tavily.com), chosen because its JSON results
are already shaped for feeding straight back into an LLM turn."""
from __future__ import annotations

from typing import Any

import httpx

from app.core.errors import ToolExecutionError, ToolUnavailableError

WEB_SEARCH_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information and return a short list of relevant results (title, url, snippet).",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "max_results": {
                    "type": "integer",
                    "description": "Max number of results to return (default 5, max 10).",
                    "minimum": 1, "maximum": 10,
                },
            },
            "required": ["query"],
        },
    },
}


class WebSearchTool:
    name = "web_search"
    schema = WEB_SEARCH_TOOL_SCHEMA

    def __init__(self, api_key: str | None, base_url: str = "https://api.tavily.com", timeout_seconds: float = 10.0):
        self._api_key = api_key
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout_seconds)

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    async def execute(self, arguments: dict[str, Any]) -> str:
        if not self.configured:
            raise ToolUnavailableError("web_search is not configured (missing API key)")
        query = arguments.get("query")
        if not query or not isinstance(query, str):
            raise ToolExecutionError("web_search called without a 'query' argument")
        try:
            max_results = max(1, min(int(arguments.get("max_results") or 5), 10))
        except (TypeError, ValueError):
            max_results = 5

        try:
            resp = await self._client.post("/search", json={"api_key": self._api_key, "query": query, "max_results": max_results})
        except httpx.TimeoutException as e:
            raise ToolExecutionError(f"web_search timed out: {e}") from e
        except httpx.ConnectError as e:
            raise ToolExecutionError(f"web_search connection failed: {e}") from e

        if resp.status_code != 200:
            raise ToolExecutionError(f"web_search returned HTTP {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        results = (data.get("results") or [])[:max_results]
        if not results:
            return "No results found."
        lines = [
            f"{i + 1}. {r.get('title', '')} ({r.get('url', '')})\n   {(r.get('content') or '')[:400]}"
            for i, r in enumerate(results)
        ]
        return "\n".join(lines)

    async def close(self) -> None:
        await self._client.aclose()
