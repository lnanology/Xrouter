"""Web Fetch tool (Tier 2 gap #4 -- the "browser/web-AI adapter" README's
"What's not implemented yet" section named but never specified): reads
one specific page's content in full, the natural complement to
web_search (which finds URLs but only returns short snippets). Backed by
Tavily's Extract API (https://docs.tavily.com/documentation/api-reference/
endpoint/extract) -- the same provider web_search already integrates, so
this needs zero new dependency and zero new secret: TAVILY_API_KEY
already covers both. Not a real browser -- there's no JS execution and no
screenshots here, Tavily's own extraction handles page rendering
server-side -- a deliberate choice over adding a Playwright dependency
(and its browser binaries) for a single optional tool; see README.md's
Tools section for the full tradeoff."""
from __future__ import annotations

from typing import Any

import httpx

from app.core.errors import ToolExecutionError, ToolUnavailableError

# A full page's extracted content can be much larger than a search
# snippet; bounded the same way every other tool-result/context
# truncation in this codebase is (web_search's per-result [:400],
# quality_gate's _MAX_JUDGED_CHARS, memory's _MEMORY_SUMMARY_MAX_CHARS).
_MAX_CONTENT_CHARS = 8000

WEB_FETCH_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_fetch",
        "description": "Fetch the full content of a specific web page URL and return it as text.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL of the page to fetch."},
            },
            "required": ["url"],
        },
    },
}


class WebFetchTool:
    name = "web_fetch"
    schema = WEB_FETCH_TOOL_SCHEMA

    def __init__(self, api_key: str | None, base_url: str = "https://api.tavily.com", timeout_seconds: float = 15.0):
        self._api_key = api_key
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout_seconds)

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    async def execute(self, arguments: dict[str, Any]) -> str:
        if not self.configured:
            raise ToolUnavailableError("web_fetch is not configured (missing API key)")
        url = arguments.get("url")
        if not url or not isinstance(url, str):
            raise ToolExecutionError("web_fetch called without a 'url' argument")

        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            resp = await self._client.post("/extract", json={"urls": url, "format": "markdown"}, headers=headers)
        except httpx.TimeoutException as e:
            raise ToolExecutionError(f"web_fetch timed out: {e}") from e
        except httpx.ConnectError as e:
            raise ToolExecutionError(f"web_fetch connection failed: {e}") from e

        if resp.status_code != 200:
            raise ToolExecutionError(f"web_fetch returned HTTP {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        results = data.get("results") or []
        if results:
            content = (results[0].get("raw_content") or "")[:_MAX_CONTENT_CHARS]
            if not content:
                return f"The page at {url} returned no extractable content."
            return f"Content from {url}:\n\n{content}"

        # Tavily reports a per-URL failure (blocked, 404, timeout on their
        # side, ...) in failed_results even on an overall HTTP 200 -- both
        # arrays must be checked, results being empty doesn't by itself
        # mean the request failed.
        failed = data.get("failed_results") or []
        if failed:
            raise ToolExecutionError(f"web_fetch could not extract {url}: {failed[0].get('error', 'unknown error')}")

        raise ToolExecutionError(f"web_fetch got no result for {url}")

    async def close(self) -> None:
        await self._client.aclose()
