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


class QueryResponse(BaseModel):
    """Response payload for POST /query."""

    answer: str
    sources: list[str] = Field(default_factory=list)
