"""Unit tests for the MCP tool client adapter, run against in-process MCP servers."""

import asyncio
from types import TracebackType
from typing import Any

import pytest
from mcp import MCPError
from mcp.server.mcpserver import MCPServer
from mcp.types import INTERNAL_ERROR, REQUEST_TIMEOUT

from insightflow.core.exceptions import ToolError, ToolTimeoutError
from insightflow.tools.calculator_server import TOOL_NAME, create_calculator_server
from insightflow.tools.mcp_client import McpToolClient
from insightflow.tools.protocols import ToolClient
from insightflow.tools.validation import ERROR_DETAIL_MAX_CHARS, argument_problem


def calculator_client(**options: Any) -> McpToolClient:
    """Build the adapter as agent nodes see it: through the `ToolClient` contract."""
    client: ToolClient = McpToolClient(
        create_calculator_server(), server_name="calculator", **options
    )
    assert isinstance(client, McpToolClient)
    return client


class FailingTransport:
    """A transport whose connection always fails, like a server that cannot be reached."""

    def __init__(self, error: BaseException) -> None:
        self.error = error

    async def __aenter__(self) -> Any:
        raise self.error

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


def build_test_server() -> tuple[MCPServer, list[str]]:
    """A server with slow, large, and failing tools; `calls` records each executed tool."""
    server = MCPServer(name="test-tools")
    calls: list[str] = []

    @server.tool()
    def echo(text: str) -> str:
        calls.append("echo")
        return text

    @server.tool()
    async def slow(text: str) -> str:
        calls.append("slow")
        await asyncio.sleep(5)
        return text

    @server.tool()
    def crash(text: str) -> str:
        calls.append("crash")
        raise RuntimeError("secret internal detail")

    @server.tool()
    def blob(size: int) -> dict[str, str]:
        calls.append("blob")
        return {"data": "x" * size}

    return server, calls


async def test_calculator_call_returns_text_and_structured_output() -> None:
    """A successful call carries the tool's text, structured data, and audit fields."""
    client = calculator_client()

    result = await client.call_tool(TOOL_NAME, {"expression": "(120 + 45) * 0.15"})

    assert result.status == "ok"
    assert result.error is None
    assert result.structured == {
        "expression": "(120 + 45) * 0.15",
        "value": pytest.approx(24.75),
        "display": "24.75",
    }
    assert "24.75" in result.output
    assert result.server == "calculator"
    assert result.tool == TOOL_NAME
    assert result.arguments == {"expression": "(120 + 45) * 0.15"}
    assert len(result.call_id) == 32
    assert result.started_at.tzinfo is not None
    assert result.duration_ms >= 0
    assert result.truncated is False


async def test_calculator_error_is_a_tool_failed_result() -> None:
    """An evaluation error comes back as data the model can use, not as an exception."""
    result = await calculator_client().call_tool(TOOL_NAME, {"expression": "1 / 0"})

    assert result.status == "error"
    assert result.error == "tool_failed"
    assert "division by zero" in result.output
    assert result.structured is None


async def test_list_tools_returns_the_calculator_schema() -> None:
    """The listed spec carries the input schema with its length limit."""
    specs = await calculator_client().list_tools()

    assert [spec.name for spec in specs] == [TOOL_NAME]
    properties = specs[0].input_schema["properties"]
    assert properties["expression"]["maxLength"] == 200
    assert specs[0].input_schema["required"] == ["expression"]


async def test_tool_list_is_fetched_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tool list is cached, so later calls open only the call connection."""
    client = calculator_client()
    opened = 0
    original = client._client

    def counting_client() -> Any:
        nonlocal opened
        opened += 1
        return original()

    monkeypatch.setattr(client, "_client", counting_client)

    await client.call_tool(TOOL_NAME, {"expression": "1 + 1"})
    await client.call_tool(TOOL_NAME, {"expression": "2 + 2"})
    await client.list_tools()

    assert opened == 3  # one list connection, then one connection per call


@pytest.mark.parametrize(
    ("arguments", "reason"),
    [
        ({}, "'expression' is a required property"),
        ({"expression": 42}, "expression: 42 is not of type 'string'"),
        ({"expression": "1+" * 150}, "expression: "),
    ],
)
async def test_invalid_arguments_get_a_short_reason(
    arguments: dict[str, Any], reason: str
) -> None:
    """A schema mismatch names the field and stays within the detail length limit."""
    result = await calculator_client().call_tool(TOOL_NAME, arguments)

    assert result.error == "invalid_arguments"
    assert result.output.startswith("Invalid arguments: ")
    assert reason in result.output
    assert len(result.output) <= ERROR_DETAIL_MAX_CHARS


async def test_schema_rejection_sends_no_call() -> None:
    """An invalid argument type stops the call on the client side."""
    server, calls = build_test_server()
    client = McpToolClient(server, server_name="test")

    result = await client.call_tool("echo", {"text": 5})

    assert result.error == "invalid_arguments"
    assert calls == []


async def test_tools_outside_the_allowlist_are_hidden_and_refused() -> None:
    """The allowlist limits both the listed tools and the tools that can be called."""
    server, calls = build_test_server()
    client = McpToolClient(server, server_name="test", allowed_tools={"echo"})

    specs = await client.list_tools()
    result = await client.call_tool("crash", {"text": "x"})

    assert [spec.name for spec in specs] == ["echo"]
    assert result.error == "unknown_tool"
    assert calls == []


async def test_unknown_tool_is_refused() -> None:
    """A tool name that the server does not offer gives `unknown_tool`."""
    result = await calculator_client().call_tool("missing", {})

    assert result.error == "unknown_tool"
    assert "missing" in result.output


async def test_server_crash_detail_is_not_exposed() -> None:
    """An unexpected server exception becomes `tool_failed` without its message."""
    server, _ = build_test_server()

    result = await McpToolClient(server, server_name="test").call_tool("crash", {"text": "x"})

    assert result.error == "tool_failed"
    assert "secret" not in result.output


async def test_slow_tool_times_out() -> None:
    """The timeout covers the call, and the result reports `timeout`."""
    server, calls = build_test_server()
    client = McpToolClient(server, server_name="test", timeout_seconds=0.3)

    result = await client.call_tool("slow", {"text": "x"})

    assert result.error == "timeout"
    assert calls == ["slow"]
    assert result.duration_ms < 3000


async def test_large_output_is_truncated() -> None:
    """Text over the limit is cut; structured data has its own, larger limit."""
    server, _ = build_test_server()
    client = McpToolClient(server, server_name="test", max_output_chars=100)

    result = await client.call_tool("blob", {"size": 500})

    assert result.status == "ok"
    assert result.truncated is True
    assert len(result.output) == 100
    assert result.structured == {"data": "x" * 500}


async def test_structured_output_over_its_limit_is_dropped() -> None:
    """Cut JSON cannot be read, so structured data over its limit is dropped, not cut."""
    server, _ = build_test_server()
    client = McpToolClient(
        server, server_name="test", max_output_chars=100, max_structured_chars=200
    )

    result = await client.call_tool("blob", {"size": 500})

    assert result.truncated is True
    assert result.structured is None


async def test_output_within_the_limit_is_kept() -> None:
    """Small results keep both text and structured data."""
    server, _ = build_test_server()
    client = McpToolClient(server, server_name="test", max_output_chars=100)

    result = await client.call_tool("blob", {"size": 10})

    assert result.truncated is False
    assert result.structured == {"data": "x" * 10}


async def test_unreachable_server_gives_unavailable_results() -> None:
    """Connection failures become `unavailable` without the transport's message."""
    client = McpToolClient(
        lambda: FailingTransport(ConnectionRefusedError("secret host 10.0.0.5")),
        server_name="remote",
    )

    result = await client.call_tool("search", {"query": "x"})

    assert result.error == "unavailable"
    assert "secret" not in result.output
    with pytest.raises(ToolError):
        await client.list_tools()


async def test_list_timeout_raises_tool_timeout() -> None:
    """A connection that does not answer in time raises `ToolTimeoutError` when listing."""

    class HangingTransport(FailingTransport):
        async def __aenter__(self) -> Any:
            await asyncio.sleep(5)

    client = McpToolClient(
        lambda: HangingTransport(RuntimeError()), server_name="remote", timeout_seconds=0.2
    )

    with pytest.raises(ToolTimeoutError):
        await client.list_tools()
    result = await client.call_tool("search", {"query": "x"})
    assert result.error == "timeout"


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (ConnectionResetError("secret detail"), "unavailable"),
        (MCPError(REQUEST_TIMEOUT, "secret detail"), "timeout"),
        (MCPError(INTERNAL_ERROR, "secret detail"), "unavailable"),
    ],
)
async def test_call_failures_after_listing_are_mapped(error: Exception, code: str) -> None:
    """A server that fails after its tools were listed gives a sanitized error result."""
    client = calculator_client()
    await client.list_tools()
    client._target = lambda: FailingTransport(error)

    result = await client.call_tool(TOOL_NAME, {"expression": "1 + 1"})

    assert result.error == code
    assert "secret" not in result.output


@pytest.mark.parametrize(
    ("error", "exception"),
    [
        (MCPError(REQUEST_TIMEOUT, "secret detail"), ToolTimeoutError),
        (MCPError(INTERNAL_ERROR, "secret detail"), ToolError),
    ],
)
async def test_protocol_errors_while_listing_raise_tool_errors(
    error: Exception, exception: type[ToolError]
) -> None:
    """MCP protocol errors during listing become the matching application error."""
    client = McpToolClient(lambda: FailingTransport(error), server_name="remote")

    with pytest.raises(exception) as raised:
        await client.list_tools()
    assert "secret" not in str(raised.value)


@pytest.mark.parametrize(
    "options",
    [
        {"server_name": " "},
        {"server_name": "s", "timeout_seconds": 0},
        {"server_name": "s", "max_output_chars": 0},
        {"server_name": "s", "max_structured_chars": 0},
    ],
)
def test_invalid_client_options_are_rejected(options: dict[str, Any]) -> None:
    """Configuration mistakes fail when the client is built."""
    with pytest.raises(ValueError):
        McpToolClient(create_calculator_server(), **options)


def test_invalid_schema_is_reported() -> None:
    """A server schema that is not valid JSON Schema blocks the call."""
    assert argument_problem({"type": "not-a-type"}, {}) == "The tool's input schema is not valid."
