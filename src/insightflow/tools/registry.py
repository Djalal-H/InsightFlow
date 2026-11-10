"""A set of tool servers that agent nodes use as one catalog."""

import logging
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from insightflow.core.exceptions import ToolError
from insightflow.tools.models import ToolResult, ToolSpec, WebSource
from insightflow.tools.protocols import ToolClient

logger = logging.getLogger(__name__)

SourceParser = Callable[[ToolResult], list[WebSource]]


@dataclass(frozen=True)
class ToolRegistration:
    """One tool server and how the set treats its tools."""

    client: ToolClient
    fixed_arguments: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    """Per tool: arguments that replace what the model asked for, for example result limits."""
    source_parser: SourceParser | None = None
    """Reads web pages from a successful result of this server's tools."""


class ToolSet:
    """Join the tools of several servers into one catalog and send each call to its server.

    A server whose tools cannot be listed is left out of the catalog for that listing, so one
    unusable server does not disable the others. When two servers offer the same tool name, the
    server registered first keeps it.
    """

    def __init__(self, registrations: Sequence[ToolRegistration] = ()) -> None:
        self._registrations = list(registrations)
        self._owners: dict[str, ToolRegistration] = {}

    @property
    def is_empty(self) -> bool:
        return not self._registrations

    async def list_tools(self) -> list[ToolSpec]:
        specs: list[ToolSpec] = []
        owners: dict[str, ToolRegistration] = {}
        for registration in self._registrations:
            server = registration.client.server_name
            try:
                server_specs = await registration.client.list_tools()
            except ToolError:
                logger.warning("tool server left out of the catalog server=%s", server)
                continue
            for spec in server_specs:
                if spec.name in owners:
                    logger.warning(
                        "duplicate tool name ignored server=%s tool=%s", server, spec.name
                    )
                    continue
                owners[spec.name] = registration
                specs.append(spec)
        self._owners = owners
        return specs

    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> ToolResult:
        """Call a tool from the last listing. Like `ToolClient.call_tool`, this never raises."""
        registration = self._owners.get(name)
        if registration is None:
            return unknown_tool_result(name, arguments)
        merged = {**arguments, **registration.fixed_arguments.get(name, {})}
        result = await registration.client.call_tool(name, merged)
        if result.status == "ok" and registration.source_parser is not None:
            result = result.model_copy(update={"web_sources": registration.source_parser(result)})
        return result


def unknown_tool_result(name: str, arguments: Mapping[str, Any]) -> ToolResult:
    """The result for a tool that no listed server offers; no server is contacted."""
    return ToolResult(
        call_id=uuid.uuid4().hex,
        server="",
        tool=name,
        arguments=dict(arguments),
        status="error",
        error="unknown_tool",
        output=f"Tool {name!r} is not available.",
        started_at=datetime.now(UTC),
        duration_ms=0.0,
    )
