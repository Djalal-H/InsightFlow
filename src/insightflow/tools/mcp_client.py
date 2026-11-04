"""`ToolClient` adapter for one MCP server, built on the official MCP SDK.

MCP SDK types stay inside this module. Agent nodes see only `ToolSpec` and `ToolResult`.
"""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Collection, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import jsonschema
from jsonschema.exceptions import best_match
from mcp import Client, MCPError, StdioServerParameters
from mcp.client import Transport
from mcp.server.mcpserver import MCPServer
from mcp.types import REQUEST_TIMEOUT, CallToolResult, TextContent

from insightflow.core.exceptions import ToolError, ToolTimeoutError
from insightflow.tools.models import ToolErrorCode, ToolResult, ToolSpec

logger = logging.getLogger(__name__)

McpServerTarget = MCPServer | StdioServerParameters | Transport | str
"""Where the server is: an in-process server, a subprocess over stdio, a transport, or a URL."""

ERROR_DETAIL_MAX_CHARS = 300
UNAVAILABLE_MESSAGE = "The tool server is not available."


class McpToolClient:
    """Call allow-listed tools on one MCP server with schema, time, and size limits.

    Each operation opens its own MCP connection and closes it at the end, so the client holds no
    open session between calls. The tool list is fetched once and cached; it gives the input
    schemas that arguments are checked against before a call is sent.
    """

    def __init__(
        self,
        target: McpServerTarget,
        *,
        server_name: str,
        allowed_tools: Collection[str] | None = None,
        timeout_seconds: float = 10.0,
        max_output_chars: int = 4000,
    ) -> None:
        if not server_name.strip():
            raise ValueError("server_name must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_output_chars < 1:
            raise ValueError("max_output_chars must be at least 1")
        self._target = target
        self._server_name = server_name
        self._allowed = frozenset(allowed_tools) if allowed_tools is not None else None
        self._timeout = timeout_seconds
        self._max_output_chars = max_output_chars
        self._specs: dict[str, ToolSpec] | None = None

    @property
    def server_name(self) -> str:
        return self._server_name

    async def list_tools(self) -> Sequence[ToolSpec]:
        return list((await self._load_specs()).values())

    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> ToolResult:
        call_id = uuid.uuid4().hex
        started_at = datetime.now(UTC)
        start = time.perf_counter()
        args = dict(arguments)

        def finish(
            *,
            error: ToolErrorCode | None = None,
            output: str = "",
            structured: dict[str, Any] | None = None,
            truncated: bool = False,
        ) -> ToolResult:
            result = ToolResult(
                call_id=call_id,
                server=self._server_name,
                tool=name,
                arguments=args,
                status="ok" if error is None else "error",
                error=error,
                output=output,
                structured=structured,
                truncated=truncated,
                started_at=started_at,
                duration_ms=(time.perf_counter() - start) * 1000,
            )
            # Arguments and output can hold user text, so only the outcome is logged.
            logger.info(
                "tool call server=%s tool=%s status=%s error=%s duration_ms=%.1f",
                result.server,
                result.tool,
                result.status,
                result.error,
                result.duration_ms,
            )
            return result

        if self._allowed is not None and name not in self._allowed:
            return finish(error="unknown_tool", output=f"Tool {name!r} is not available.")

        try:
            specs = await self._load_specs()
        except ToolTimeoutError:
            return finish(error="timeout", output="The tool server did not respond in time.")
        except ToolError:
            return finish(error="unavailable", output=UNAVAILABLE_MESSAGE)

        spec = specs.get(name)
        if spec is None:
            return finish(error="unknown_tool", output=f"Tool {name!r} is not available.")
        problem = argument_problem(spec.input_schema, args)
        if problem is not None:
            return finish(error="invalid_arguments", output=problem)

        try:
            async with asyncio.timeout(self._timeout), self._client() as client:
                result = await client.call_tool(name, args)
        except TimeoutError:
            return finish(error="timeout", output="The tool did not finish in time.")
        except MCPError as exc:
            if exc.code == REQUEST_TIMEOUT:
                return finish(error="timeout", output="The tool did not finish in time.")
            self._log_failure("call", exc)
            return finish(error="unavailable", output=UNAVAILABLE_MESSAGE)
        except Exception as exc:  # noqa: BLE001 - transports raise many unrelated types
            self._log_failure("call", exc)
            return finish(error="unavailable", output=UNAVAILABLE_MESSAGE)

        text, structured, truncated = self._limit_output(result)
        if result.is_error:
            return finish(error="tool_failed", output=_clip(text or "The tool reported an error."))
        return finish(output=text, structured=structured, truncated=truncated)

    def _client(self) -> Client:
        # The MCP response cache is off: specs are cached here, and tool results must be fresh.
        return Client(self._target, cache=None)

    async def _load_specs(self) -> dict[str, ToolSpec]:
        if self._specs is not None:
            return self._specs
        try:
            async with asyncio.timeout(self._timeout), self._client() as client:
                specs: dict[str, ToolSpec] = {}
                cursor: str | None = None
                while True:
                    page = await client.list_tools(cursor=cursor)
                    for tool in page.tools:
                        if self._allowed is None or tool.name in self._allowed:
                            specs[tool.name] = ToolSpec(
                                name=tool.name,
                                description=tool.description or "",
                                input_schema=dict(tool.input_schema),
                            )
                    cursor = page.next_cursor
                    if cursor is None:
                        break
        except TimeoutError as exc:
            raise ToolTimeoutError from exc
        except MCPError as exc:
            if exc.code == REQUEST_TIMEOUT:
                raise ToolTimeoutError from exc
            self._log_failure("list", exc)
            raise ToolError from exc
        except Exception as exc:  # noqa: BLE001 - transports raise many unrelated types
            self._log_failure("list", exc)
            raise ToolError from exc
        self._specs = specs
        return specs

    def _log_failure(self, operation: str, exc: Exception) -> None:
        # The exception text can hold server or transport details, so only its type is logged.
        if isinstance(exc, MCPError):
            detail = f"code={exc.code}"
        else:
            detail = f"error_type={type(exc).__name__}"
        logger.warning("tool %s failed server=%s %s", operation, self._server_name, detail)

    def _limit_output(self, result: CallToolResult) -> tuple[str, dict[str, Any] | None, bool]:
        """Join the text content and enforce the output size limit."""
        structured = result.structured_content
        text = "\n".join(
            block.text for block in result.content if isinstance(block, TextContent)
        )
        if not text and structured is not None:
            text = json.dumps(structured, ensure_ascii=False)
        truncated = False
        if len(text) > self._max_output_chars:
            text = text[: self._max_output_chars]
            truncated = True
        if (
            structured is not None
            and len(json.dumps(structured, ensure_ascii=False)) > self._max_output_chars
        ):
            structured = None
            truncated = True
        return text, structured, truncated


def argument_problem(schema: dict[str, Any], arguments: Mapping[str, Any]) -> str | None:
    """Return a short reason when the arguments do not match the JSON schema, else None."""
    try:
        validator_class = jsonschema.validators.validator_for(schema)
        validator_class.check_schema(schema)
    except jsonschema.SchemaError:
        return "The tool's input schema is not valid."
    error = best_match(validator_class(schema).iter_errors(dict(arguments)))
    if error is None:
        return None
    location = ".".join(str(part) for part in error.absolute_path)
    reason = f"{location}: {error.message}" if location else error.message
    return _clip(f"Invalid arguments: {reason}")


def _clip(text: str) -> str:
    if len(text) <= ERROR_DETAIL_MAX_CHARS:
        return text
    return text[: ERROR_DETAIL_MAX_CHARS - 3] + "..."
