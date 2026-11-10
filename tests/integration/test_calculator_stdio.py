"""The calculator server runs as a subprocess and answers over the MCP stdio transport."""

import sys
from pathlib import Path

from mcp import StdioServerParameters

from insightflow.tools.calculator_server import TOOL_NAME
from insightflow.tools.mcp_client import McpToolClient

REPO_ROOT = Path(__file__).resolve().parents[2]


async def test_calculator_over_stdio() -> None:
    """A real subprocess lists the tool and returns a calculation through `McpToolClient`."""
    target = StdioServerParameters(
        command=sys.executable,
        args=["-m", "insightflow.tools.calculator_server"],
        env={"PYTHONPATH": str(REPO_ROOT / "src")},
        cwd=REPO_ROOT,
    )
    client = McpToolClient(target, server_name="calculator", timeout_seconds=30)

    specs = await client.list_tools()
    result = await client.call_tool(TOOL_NAME, {"expression": "2 ** 10 + 1"})

    assert [spec.name for spec in specs] == [TOOL_NAME]
    assert result.status == "ok", result.output
    assert result.structured is not None
    assert result.structured["display"] == "1025"
