"""Boundary between agent nodes and tool servers."""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from insightflow.tools.models import ToolResult, ToolSpec


class ToolClient(Protocol):
    """Contract used by agent nodes that call tools on one server."""

    @property
    def server_name(self) -> str: ...

    async def list_tools(self) -> Sequence[ToolSpec]:
        """Return the allowed tools. Raise `ToolError` when the server is unusable."""
        ...

    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> ToolResult:
        """Call one tool. Failures are returned as results with an error code, not raised."""
        ...
