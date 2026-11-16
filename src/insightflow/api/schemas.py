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


QUERY_MAX_CHARS = 4000
"""Longer questions are refused (422): a length limit is the first input screen (OWASP)."""


class QueryFilters(BaseModel):
    """Metadata filters that limit every document search of the request."""

    file_types: list[Literal["pdf", "docx", "md", "txt"]] | None = Field(
        default=None, min_length=1, description="Search only documents of these file types."
    )
    sources: list[Annotated[str, StringConstraints(min_length=1, max_length=512)]] | None = Field(
        default=None,
        min_length=1,
        max_length=50,
        description=(
            "Search only these documents, by corpus-relative path as `source_identifier` "
            "reports it, for example `PDF/student_handbook.pdf`."
        ),
    )


class QueryRequest(BaseModel):
    """Request payload for POST /query and POST /query/stream."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=QUERY_MAX_CHARS,
        description="The user's natural-language question.",
    )
    filters: QueryFilters | None = Field(
        default=None, description="Optional limits for document search. Web search ignores them."
    )
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
        Literal[
            "unknown_tool",
            "invalid_arguments",
            "tool_failed",
            "timeout",
            "unavailable",
            "not_authorized",
            "not_approved",
        ]
        | None
    ) = None
    truncated: bool = Field(False, description="The tool output was cut to the size limit.")
    sources: list[int] = Field(
        default_factory=list, description="Citation numbers of the sources that the step returned."
    )
    duration_ms: float = Field(..., ge=0)


class ProviderUsage(BaseModel):
    """Calls to one kind of hosted model during the request."""

    calls: int = Field(default=0, ge=0)
    errors: int = Field(default=0, ge=0)
    prompt_tokens: int | None = Field(
        default=None, description="Sum over the calls that reported usage; null when none did."
    )
    completion_tokens: int | None = None
    unreported_usage: int = Field(
        default=0, ge=0, description="Calls for which the provider reported no token use."
    )


class UsageReport(BaseModel):
    """Cost and reliability counters of the request."""

    chat: ProviderUsage = Field(default_factory=lambda: ProviderUsage())
    embedding: ProviderUsage = Field(default_factory=lambda: ProviderUsage())
    rerank: ProviderUsage = Field(default_factory=lambda: ProviderUsage())
    store_calls: int = Field(default=0, ge=0, description="Vector-store operations.")
    retrieval_calls: int = Field(default=0, ge=0, description="search_documents steps.")
    retrieved_passages: int = Field(
        default=0, ge=0, description="Passages returned by those steps."
    )
    tool_calls: int = Field(
        default=0, ge=0, description="Steps that called a tool other than search."
    )
    tool_errors: int = Field(
        default=0, ge=0, description="Steps of any tool that ended in an error."
    )
    retries: int = Field(default=0, ge=0, description="Answer attempts after the first.")
    errors: dict[str, int] = Field(
        default_factory=dict, description="Recorded failures by stable error code."
    )
    node_timings_ms: dict[str, float] = Field(
        default_factory=dict, description="Total time per workflow node."
    )


class ExecutionMetadata(BaseModel):
    """What the workflow did to produce the answer."""

    attempts: int = Field(..., ge=1)
    retrieval_queries: list[str] = Field(
        default_factory=list, description="The queries of every search_documents step."
    )
    steps: list[StepReport] = Field(default_factory=list, description="Agent actions in order.")
    nodes: list[str] = Field(default_factory=list, description="Workflow nodes in run order.")
    recalled_turns: int = Field(
        0, ge=0, description="Older turns that conversation memory added to the prompts."
    )
    duration_ms: int = Field(..., ge=0)
    usage: UsageReport = Field(default_factory=lambda: UsageReport())
    guardrails: list[str] = Field(
        default_factory=list,
        description=(
            "What the guardrails found or changed, as `<where>:<signal>` codes, for example "
            "`query:instruction_override`, `web:hidden_text`, or `answer:external_link`."
        ),
    )


class PendingApproval(BaseModel):
    """A consequential tool call that waits for the user's decision."""

    approval_id: str = Field(..., description="Send it back with the decision.")
    tool: str
    server: str
    arguments: dict[str, Any] = Field(..., description="Exactly what the tool will receive.")
    thought: str = Field(default="", description="The agent's short reason for the call.")


class QueryResponse(BaseModel):
    """Response payload for POST /query, and the final event of POST /query/stream."""

    status: Literal["completed", "approval_required"] = Field(
        default="completed",
        description=(
            "approval_required: the run is paused before a consequential tool call; send a "
            "decision to POST /threads/{thread_id}/approvals/{approval_id}. `answer` is empty."
        ),
    )
    answer: str
    sources: list[Source] = Field(default_factory=list)
    thread_id: str | None = Field(
        default=None, description="The conversation thread, or null when storage is disabled."
    )
    verification: VerificationReport
    metadata: ExecutionMetadata
    pending_approval: PendingApproval | None = None


class ApprovalDecision(BaseModel):
    """Request payload for POST /threads/{thread_id}/approvals/{approval_id}."""

    decision: Literal["approve", "reject"]


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
        "unauthorized",
        "forbidden",
        "rate_limited",
        "quota_exceeded",
        "rate_limiter_unavailable",
        "input_rejected",
        "approval_not_found",
        "payload_too_large",
    ]
    message: str


class ErrorResponse(BaseModel):
    """Shared HTTP error envelope."""

    error: ErrorDetail
