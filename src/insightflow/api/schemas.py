"""Shared HTTP response schemas."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, StringConstraints


class HealthResponse(BaseModel):
    """Health endpoint response."""

    status: Literal["ok", "not_ready"]
    checks: dict[str, str] = Field(default_factory=dict)


ThreadId = Annotated[
    str, StringConstraints(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
]


class QueryRequest(BaseModel):
    """Request payload for POST /query and POST /query/stream."""

    query: str = Field(..., min_length=1, description="The user's natural-language question.")
    thread_id: ThreadId | None = Field(
        default=None,
        description=(
            "Conversation thread to continue. Omit it to start a new thread; the response "
            "returns the new identifier. Ignored when conversation storage is disabled."
        ),
    )


class SourceReference(BaseModel):
    """One context passage given to the model, numbered as cited in the answer."""

    kind: Literal["document"] = "document"
    citation: int = Field(..., ge=1, description="The passage number used in answer citations.")
    source_identifier: str
    source_name: str
    chunk_id: str
    page_start: int | None = None
    page_end: int | None = None
    score: float


class WebSourceReference(BaseModel):
    """One web page returned by a search tool, numbered as cited in the answer."""

    kind: Literal["web"] = "web"
    citation: int = Field(..., ge=1, description="The result number used in answer citations.")
    title: str
    url: str


Source = Annotated[SourceReference | WebSourceReference, Field(discriminator="kind")]


class VerificationReport(BaseModel):
    """How the answer was checked against its context."""

    status: Literal["passed", "failed", "unchecked", "skipped"] = Field(
        ...,
        description=(
            "passed: checks found no problem. failed: checks found problems after every "
            "allowed attempt. unchecked: the judge response could not be read. skipped: "
            "there were no sources to check against (direct replies, answers from tool results "
            "only, and searches that found nothing)."
        ),
    )
    method: Literal["none", "deterministic", "llm"]
    issues: list[str] = Field(default_factory=list)


class StepReport(BaseModel):
    """Audit record of one agent action: why it was taken, the call, and its outcome."""

    attempt: int = Field(..., ge=1)
    thought: str = Field("", description="The agent's short reason for the action.")
    tool: str
    server: str = Field(..., description="`insightflow` for document search, else the MCP server.")
    arguments: dict[str, Any]
    call_id: str
    status: Literal["ok", "error"]
    error: (
        Literal["unknown_tool", "invalid_arguments", "tool_failed", "timeout", "unavailable"]
        | None
    ) = None
    truncated: bool = Field(False, description="The tool output was cut to the size limit.")
    sources: list[int] = Field(
        default_factory=list, description="Citation numbers of the sources that the step returned."
    )
    duration_ms: float = Field(..., ge=0)


class ExecutionMetadata(BaseModel):
    """What the workflow did to produce the answer."""

    attempts: int = Field(..., ge=1)
    retrieval_queries: list[str] = Field(
        default_factory=list, description="The queries of every search_documents step."
    )
    steps: list[StepReport] = Field(default_factory=list, description="Agent actions in order.")
    nodes: list[str] = Field(default_factory=list, description="Workflow nodes in run order.")
    duration_ms: int = Field(..., ge=0)


class QueryResponse(BaseModel):
    """Response payload for POST /query, and the final event of POST /query/stream."""

    answer: str
    sources: list[Source] = Field(default_factory=list)
    thread_id: str | None = Field(
        default=None, description="The conversation thread, or null when storage is disabled."
    )
    verification: VerificationReport
    metadata: ExecutionMetadata


class ConversationTurnResponse(BaseModel):
    """One completed question and answer."""

    query: str
    answer: str


class ThreadResponse(BaseModel):
    """Response payload for GET /threads/{thread_id}."""

    thread_id: str
    turns: list[ConversationTurnResponse]


class ErrorDetail(BaseModel):
    """Stable, sanitized description of an expected application failure."""

    code: Literal[
        "provider_configuration_error",
        "provider_timeout",
        "provider_rate_limited",
        "provider_error",
        "storage_unavailable",
        "thread_not_found",
    ]
    message: str


class ErrorResponse(BaseModel):
    """Shared HTTP error envelope."""

    error: ErrorDetail
