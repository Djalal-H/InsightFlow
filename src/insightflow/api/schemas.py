"""Shared HTTP response schemas."""

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health endpoint response."""

    status: Literal["ok", "not_ready"]
    checks: dict[str, str] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    """Request payload for POST /query."""

    query: str = Field(..., min_length=1, description="The user's natural-language question.")


class SourceReference(BaseModel):
    """One context passage given to the model, numbered as cited in the answer."""

    citation: int = Field(..., ge=1, description="The passage number used in answer citations.")
    source_identifier: str
    source_name: str
    chunk_id: str
    page_start: int | None = None
    page_end: int | None = None
    score: float


class QueryResponse(BaseModel):
    """Response payload for POST /query."""

    answer: str
    sources: list[SourceReference] = Field(default_factory=list)


class ErrorDetail(BaseModel):
    """Stable, sanitized description of an expected application failure."""

    code: Literal[
        "provider_configuration_error",
        "provider_timeout",
        "provider_rate_limited",
        "provider_error",
        "storage_unavailable",
    ]
    message: str


class ErrorResponse(BaseModel):
    """Shared HTTP error envelope."""

    error: ErrorDetail
