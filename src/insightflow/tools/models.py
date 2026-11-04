"""Provider-neutral tool models shared by tool clients and agent nodes."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ToolCallStatus = Literal["ok", "error"]

ToolErrorCode = Literal[
    "unknown_tool",
    "invalid_arguments",
    "tool_failed",
    "timeout",
    "unavailable",
]
"""Stable failure codes for a tool call.

- `unknown_tool`: the tool is not on the allowlist or the server does not offer it.
- `invalid_arguments`: the arguments do not match the tool's input schema.
- `tool_failed`: the server ran the tool and reported an error.
- `timeout`: the call did not finish before the client's timeout.
- `unavailable`: the server could not be reached or replied with an unusable result.
"""


class ToolSpec(BaseModel):
    """One tool that a server offers and the allowlist permits."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """The outcome and audit record of one tool call.

    Every call returns a result, including failed calls, so that the workflow can record the
    attempt and continue. `output` is the text given to the model: the tool's text content on
    success, or a short, length-limited reason on failure.
    """

    model_config = ConfigDict(frozen=True)

    call_id: str
    server: str
    tool: str
    arguments: dict[str, Any]
    status: ToolCallStatus
    error: ToolErrorCode | None = None
    output: str = ""
    structured: dict[str, Any] | None = None
    truncated: bool = False
    started_at: datetime
    duration_ms: float = Field(ge=0.0)
