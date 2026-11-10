"""Unit tests for the tool set, the Tavily registration, and Tavily result parsing."""

from datetime import UTC, datetime
from types import TracebackType
from typing import Any

import pytest
from mcp.server.mcpserver import MCPServer

from insightflow.api.dependencies import create_tool_set
from insightflow.core.config import Settings
from insightflow.core.exceptions import ToolError
from insightflow.tools import mcp_client
from insightflow.tools.calculator_server import TOOL_NAME as CALCULATOR_TOOL
from insightflow.tools.mcp_client import McpToolClient
from insightflow.tools.models import ToolResult, WebSource
from insightflow.tools.registry import ToolRegistration, ToolSet
from insightflow.tools.web_search import (
    TAVILY_SEARCH_TOOL,
    create_tavily_registration,
    parse_tavily_sources,
)
from tests.fakes import TAVILY_TEXT, FakeToolClient


def tool_result(output: str = "", structured: dict[str, Any] | None = None) -> ToolResult:
    return ToolResult(
        call_id="c",
        server="tavily",
        tool=TAVILY_SEARCH_TOOL,
        arguments={},
        status="ok",
        output=output,
        structured=structured,
        started_at=datetime.now(UTC),
        duration_ms=0.0,
    )


# --- ToolSet -----------------------------------------------------------------------------


async def test_tool_set_joins_servers_and_skips_unusable_ones() -> None:
    """One failing server does not hide the tools of the others; duplicates keep the first."""
    first = FakeToolClient("first", ["search", "calc"])
    broken = FakeToolClient("broken", ["other"], list_error=ToolError())
    second = FakeToolClient("second", ["calc", "weather"])
    tools = ToolSet([ToolRegistration(first), ToolRegistration(broken), ToolRegistration(second)])

    specs = await tools.list_tools()
    await tools.call_tool("calc", {"query": "1"})
    await tools.call_tool("weather", {"query": "Paris"})

    assert [spec.name for spec in specs] == ["search", "calc", "weather"]
    assert first.calls == [("calc", {"query": "1"})]
    assert second.calls == [("weather", {"query": "Paris"})]


async def test_tool_set_applies_fixed_arguments_and_reads_sources() -> None:
    """Fixed arguments replace the model's values; the parser fills `web_sources`."""
    client = FakeToolClient("tavily", ["search"], replies={"search": TAVILY_TEXT})
    tools = ToolSet(
        [
            ToolRegistration(
                client,
                fixed_arguments={"search": {"max_results": 3}},
                source_parser=parse_tavily_sources,
            )
        ]
    )
    await tools.list_tools()

    result = await tools.call_tool("search", {"query": "news", "max_results": 20})

    assert client.calls == [("search", {"query": "news", "max_results": 3})]
    assert result.arguments == {"query": "news", "max_results": 3}
    assert [source.url for source in result.web_sources] == [
        "https://fastapi.tiangolo.com/release-notes/",
        "https://docs.pydantic.dev/latest/changelog/",
    ]


async def test_tool_set_refuses_unlisted_tools_without_a_call() -> None:
    """A name that is not in the last listing gives `unknown_tool` and contacts no server."""
    client = FakeToolClient("fake", ["search"])
    tools = ToolSet([ToolRegistration(client)])
    await tools.list_tools()

    result = await tools.call_tool("delete_everything", {})

    assert result.error == "unknown_tool"
    assert result.server == ""
    assert client.calls == []


def test_empty_tool_set() -> None:
    assert ToolSet().is_empty
    assert not ToolSet([ToolRegistration(FakeToolClient())]).is_empty


# --- Tavily parsing ----------------------------------------------------------------------


def test_parse_tavily_text_format() -> None:
    """The text format of the Tavily MCP server becomes one source per page."""
    text = (
        "Answer: ignored\n\nDetailed Results:\n\n"
        "Title: First\nID: 1\nURL: https://a.example/\nContent: one\ncontinued\n"
        "Raw Content: not used\nFavicon: https://a.example/icon.png\n\n"
        "URL: https://b.example/\nContent: no title\n\n"
        "Title: Not a web page\nURL: ftp://c.example/\nContent: dropped\n\n"
        "Images:\n\n[1] URL: https://img.example/x.png"
    )

    sources = parse_tavily_sources(tool_result(text))

    assert sources == [
        WebSource(title="First", url="https://a.example/", content="one\ncontinued"),
        WebSource(title="https://b.example/", url="https://b.example/", content="no title"),
    ]


def test_parse_tavily_structured_results_are_preferred() -> None:
    structured = {
        "results": [
            {"title": "A", "url": "https://a.example/", "content": "x"},
            {"url": "https://b.example/"},
            {"title": "No URL"},
            "not a dict",
        ]
    }

    sources = parse_tavily_sources(tool_result("Title: ignored", structured))

    assert sources == [
        WebSource(title="A", url="https://a.example/", content="x"),
        WebSource(title="https://b.example/", url="https://b.example/", content=""),
    ]


TAVILY_JSON = (
    '{"query": "q", "answer": null, "images": [], "results": ['
    '{"url": "https://a.example/", "title": "A", "content": "first", "score": 0.9, '
    '"raw_content": null, "id": "a3da8f-00"}, '
    '{"url": "https://b.example/", "title": "B", "content": "second", "id": "70bd86-01"}]}'
)


def test_parse_tavily_json_text_without_structured_content() -> None:
    """The hosted server sends its results as JSON text too."""
    sources = parse_tavily_sources(tool_result(TAVILY_JSON))

    assert [(source.title, source.url, source.content) for source in sources] == [
        ("A", "https://a.example/", "first"),
        ("B", "https://b.example/", "second"),
    ]


def test_parse_tavily_cut_json_text_gives_no_sources() -> None:
    assert parse_tavily_sources(tool_result(TAVILY_JSON[:120])) == []


def test_parse_tavily_clips_page_content() -> None:
    structured = {"results": [{"title": "A", "url": "https://a.example/", "content": "y" * 50}]}

    (page,) = parse_tavily_sources(tool_result(structured=structured), max_page_chars=20)

    assert page.content == "y" * 17 + "..."


async def test_large_tavily_result_keeps_all_pages() -> None:
    """Regression: a result over the text limit lost all its pages.

    The client cut the JSON text and dropped the structured data, so the parser read nothing
    and the model saw cut JSON. The structured data now has its own limit.
    """
    server = MCPServer("fake-tavily")

    @server.tool(name=TAVILY_SEARCH_TOOL)
    def search(query: str) -> dict[str, Any]:
        pages = [
            {"title": f"Page {n}", "url": f"https://site{n}.example/", "content": "z" * 1500}
            for n in range(5)
        ]
        return {"query": query, "results": pages}

    client = McpToolClient(server, server_name="tavily", max_output_chars=6000)
    tools = ToolSet(
        [
            ToolRegistration(
                client,
                source_parser=lambda result: parse_tavily_sources(result, max_page_chars=1200),
            )
        ]
    )
    await tools.list_tools()

    result = await tools.call_tool(TAVILY_SEARCH_TOOL, {"query": "q"})

    assert result.truncated is True
    assert [page.url for page in result.web_sources] == [
        f"https://site{n}.example/" for n in range(5)
    ]
    assert all(len(page.content) == 1200 for page in result.web_sources)


def test_parse_tavily_without_results() -> None:
    assert parse_tavily_sources(tool_result("No results found.")) == []


# --- Tavily registration -----------------------------------------------------------------


class CapturingHttpClient:
    """Records the headers of each HTTP client and fails the connection."""

    headers: list[dict[str, str]] = []

    def __init__(self, *, headers: dict[str, str], timeout: Any) -> None:
        CapturingHttpClient.headers.append(headers)

    async def __aenter__(self) -> "CapturingHttpClient":
        raise ConnectionRefusedError("offline test")

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


async def test_tavily_key_is_sent_in_a_header_not_the_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each connection gets a new HTTP client with a bearer header; the URL carries no key."""
    CapturingHttpClient.headers = []
    monkeypatch.setattr(mcp_client.httpx2, "AsyncClient", CapturingHttpClient)
    registration = create_tavily_registration(
        "https://mcp.tavily.test/mcp/",
        "tvly-secret",
        max_results=4,
        timeout_seconds=5,
        max_output_chars=1000,
    )

    with pytest.raises(ToolError):
        await registration.client.list_tools()
    with pytest.raises(ToolError):
        await registration.client.list_tools()

    assert CapturingHttpClient.headers == [{"Authorization": "Bearer tvly-secret"}] * 2
    assert registration.client.server_name == "tavily"
    assert registration.fixed_arguments == {
        TAVILY_SEARCH_TOOL: {
            "max_results": 4,
            "include_raw_content": False,
            "include_images": False,
        }
    }
    assert registration.source_parser is not None
    long_page = {"results": [{"title": "A", "url": "https://a.example/", "content": "x" * 900}]}
    (page,) = registration.source_parser(tool_result(structured=long_page))
    assert len(page.content) == 1000 // 4


# --- settings and composition ------------------------------------------------------------


async def test_default_tool_set_has_only_the_calculator() -> None:
    settings = Settings(
        _env_file=None, tool_calculator_enabled=True, tool_web_search_enabled=False
    )

    specs = await create_tool_set(settings).list_tools()

    assert [spec.name for spec in specs] == [CALCULATOR_TOOL]


def test_tool_set_with_everything_disabled_is_empty() -> None:
    settings = Settings(
        _env_file=None, tool_calculator_enabled=False, tool_web_search_enabled=False
    )

    assert create_tool_set(settings).is_empty


def test_web_search_tool_set_registers_tavily_without_connecting() -> None:
    """Building the set opens no connection; servers are contacted on first use."""
    settings = Settings(
        _env_file=None,
        tool_calculator_enabled=False,
        tool_web_search_enabled=True,
        tavily_api_key="tvly-secret",
    )

    assert not create_tool_set(settings).is_empty


@pytest.mark.parametrize("key", [None, "", "   "])
def test_web_search_requires_an_api_key(key: str | None) -> None:
    with pytest.raises(ValueError, match="TAVILY_API_KEY"):
        Settings(_env_file=None, tool_web_search_enabled=True, tavily_api_key=key)


def test_tavily_key_is_hidden_in_settings_output() -> None:
    settings = Settings(_env_file=None, tavily_api_key="tvly-secret")

    assert "tvly-secret" not in repr(settings)


@pytest.mark.parametrize(
    "values",
    [
        {"agent_max_steps": 0},
        {"agent_max_steps": 9},
        {"tool_timeout_seconds": 0},
        {"tool_max_output_chars": 10},
        {"web_search_max_results": 21},
    ],
)
def test_tool_settings_are_validated(values: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        Settings(_env_file=None, **values)
