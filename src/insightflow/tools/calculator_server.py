"""InsightFlow's calculator MCP server.

InsightFlow can connect to this server in-process. Other MCP clients can run it over stdio:

    PYTHONPATH=src python -m insightflow.tools.calculator_server

`MCPServer` calls `logging.basicConfig` when it is created. That call does nothing when the
root logger already has handlers, so configure application logging before an in-process server
is created, or the SDK installs its own handler.
"""

from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from insightflow.tools.calculator import (
    MAX_EXPRESSION_CHARS,
    CalculationError,
    evaluate,
    format_number,
)

SERVER_NAME = "insightflow-calculator"
TOOL_NAME = "calculate"

TOOL_DESCRIPTION = (
    "Evaluate one arithmetic expression exactly. Use it for sums, differences, products, "
    "ratios, percentages, and averages instead of calculating mentally. Supports numbers, "
    "+ - * / // % ** and parentheses, the functions abs, round, min, max, sqrt, floor, ceil, "
    "log, log10, exp, and the constants pi and e. Write a percentage as a fraction, for "
    "example 15% of 80 as 0.15 * 80."
)


class Calculation(BaseModel):
    """Structured result of the calculate tool."""

    expression: str
    value: int | float
    display: str


def calculate(
    expression: Annotated[
        str,
        Field(
            min_length=1,
            max_length=MAX_EXPRESSION_CHARS,
            description="An arithmetic expression, for example (120 + 45) * 0.15",
        ),
    ],
) -> Calculation:
    """Evaluate the expression; failures go back to the client as tool errors."""
    try:
        value = evaluate(expression)
    except CalculationError as exc:
        raise ToolError(str(exc)) from exc
    return Calculation(expression=expression.strip(), value=value, display=format_number(value))


def create_calculator_server() -> MCPServer:
    """Build a new calculator server with one read-only tool."""
    server = MCPServer(
        name=SERVER_NAME,
        instructions="Exact arithmetic for numbers found in documents or questions.",
    )
    server.add_tool(
        calculate,
        name=TOOL_NAME,
        title="Calculator",
        description=TOOL_DESCRIPTION,
        annotations=ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    )
    return server


def main() -> None:
    """Serve the calculator over stdio."""
    create_calculator_server().run("stdio")


if __name__ == "__main__":
    main()
