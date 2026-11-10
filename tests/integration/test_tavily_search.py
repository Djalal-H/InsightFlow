"""Opt-in test against Tavily's hosted MCP server. Each run makes one billable search.

Run with `TAVILY_TEST_API_KEY=tvly-... pytest -m tavily`. The test uses its own variable, not
`TAVILY_API_KEY`, because LiteLLM loads `.env` into the environment and the default suite must
never make a billable call.
"""

import os

import pytest

from insightflow.tools.registry import ToolSet
from insightflow.tools.web_search import TAVILY_SEARCH_TOOL, create_tavily_registration

TAVILY_TEST_API_KEY = os.environ.get("TAVILY_TEST_API_KEY")
TAVILY_TEST_URL = os.environ.get("TAVILY_TEST_URL", "https://mcp.tavily.com/mcp/")

pytestmark = [
    pytest.mark.tavily,
    pytest.mark.skipif(not TAVILY_TEST_API_KEY, reason="TAVILY_TEST_API_KEY is not set"),
]


async def test_tavily_search_returns_web_sources() -> None:
    assert TAVILY_TEST_API_KEY is not None
    registration = create_tavily_registration(
        TAVILY_TEST_URL,
        TAVILY_TEST_API_KEY,
        max_results=3,
        timeout_seconds=60,
        max_output_chars=6000,
    )
    tools = ToolSet([registration])

    specs = await tools.list_tools()
    result = await tools.call_tool(TAVILY_SEARCH_TOOL, {"query": "FastAPI web framework"})

    assert [spec.name for spec in specs] == [TAVILY_SEARCH_TOOL]
    assert result.status == "ok", (result.error, result.output)
    assert result.arguments["max_results"] == 3
    assert 1 <= len(result.web_sources) <= 3
    assert all(source.url.startswith("http") for source in result.web_sources)
