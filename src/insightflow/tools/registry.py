"""A set of tool servers that agent nodes use as one catalog."""

import logging
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from insightflow.core.exceptions import ToolError
from insightflow.tools.models import ToolErrorCode, ToolResult, ToolSpec, WebSource
from insightflow.tools.protocols import ToolClient

logger = logging.getLogger(__name__)

SourceParser = Callable[[ToolResult], list[WebSource]]

DEFAULT_FAILURE_COOLDOWN_SECONDS = 60.0
PAUSING_ERRORS: frozenset[ToolErrorCode] = frozenset({"timeout", "unavailable"})
"""Call errors that show a server problem. Errors in the arguments or in the tool's own work
(`invalid_arguments`, `tool_failed`) do not pause the server."""
PAUSED_MESSAGE = "The tool server is not available. It is paused after a failure."


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

    A server whose tools cannot be listed is left out of the catalog, so one unusable server
    does not disable the others. When two servers offer the same tool name, the server
    registered first keeps it.

    A failed listing, or a call that fails with `timeout` or `unavailable`, pauses the server for
    `failure_cooldown_seconds`. While it is paused, the catalog leaves it out and its calls fail
    at once without a connection, so a server that is down does not cost a timeout on each agent
    step. The set is shared by all requests, so the pause applies to all of them. The first
    listing after the pause tries the server again. A cooldown of 0 turns pauses off.
    """

    def __init__(
        self,
        registrations: Sequence[ToolRegistration] = (),
        *,
        failure_cooldown_seconds: float = DEFAULT_FAILURE_COOLDOWN_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_cooldown_seconds < 0:
            raise ValueError("failure_cooldown_seconds must not be negative")
        self._registrations = list(registrations)
        self._owners: dict[str, ToolRegistration] = {}
        self._cooldown = failure_cooldown_seconds
        self._clock = clock
        self._paused_until: dict[int, float] = {}

    @property
    def is_empty(self) -> bool:
        return not self._registrations

    @property
    def failure_cooldown_seconds(self) -> float:
        return self._cooldown

    def is_paused(self, registration: ToolRegistration) -> bool:
        until = self._paused_until.get(id(registration))
        return until is not None and self._clock() < until

    async def list_tools(self) -> list[ToolSpec]:
        specs: list[ToolSpec] = []
        owners: dict[str, ToolRegistration] = {}
        for registration in self._registrations:
            server = registration.client.server_name
            if self.is_paused(registration):
                continue
            try:
                server_specs = await registration.client.list_tools()
            except ToolError:
                self._pause(registration, "listing failed")
                continue
            self._paused_until.pop(id(registration), None)
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
        if self.is_paused(registration):
            return _local_error(
                name, merged, "unavailable", PAUSED_MESSAGE, registration.client.server_name
            )
        result = await registration.client.call_tool(name, merged)
        if result.error in PAUSING_ERRORS:
            self._pause(registration, f"call {result.error}")
        if result.status == "ok" and registration.source_parser is not None:
            result = result.model_copy(update={"web_sources": registration.source_parser(result)})
        return result

    def _pause(self, registration: ToolRegistration, reason: str) -> None:
        server = registration.client.server_name
        if self._cooldown == 0:
            logger.warning("tool server failed server=%s reason=%s", server, reason)
            return
        self._paused_until[id(registration)] = self._clock() + self._cooldown
        logger.warning(
            "tool server paused server=%s reason=%s cooldown_s=%.0f",
            server,
            reason,
            self._cooldown,
        )


def unknown_tool_result(name: str, arguments: Mapping[str, Any]) -> ToolResult:
    """The result for a tool that no listed server offers; no server is contacted."""
    return _local_error(name, arguments, "unknown_tool", f"Tool {name!r} is not available.")


def _local_error(
    name: str,
    arguments: Mapping[str, Any],
    error: ToolErrorCode,
    output: str,
    server: str = "",
) -> ToolResult:
    """An error result made by the set itself, without a server call."""
    return ToolResult(
        call_id=uuid.uuid4().hex,
        server=server,
        tool=name,
        arguments=dict(arguments),
        status="error",
        error=error,
        output=output,
        started_at=datetime.now(UTC),
        duration_ms=0.0,
    )
