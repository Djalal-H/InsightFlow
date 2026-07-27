"""Provider-neutral domain models for ingestion and retrieval."""

from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

NonEmptyString = Annotated[str, Field(min_length=1)]
ElementType = Literal[
    "title",
    "heading",
    "paragraph",
    "list_item",
    "table",
    "caption",
    "code",
    "other",
]
IngestionStatus = Literal["completed", "partial", "rejected", "skipped"]


class DomainModel(BaseModel):
    """Strict immutable base for values crossing RAG boundaries."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DocumentSource(DomainModel):
    """One file selected from an explicitly supplied corpus directory."""

    source_name: NonEmptyString
    source_identifier: NonEmptyString
    path: Path
    mime_type: NonEmptyString
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class DocumentElement(DomainModel):
    """A normalized structural element in deterministic document order."""

    element_id: NonEmptyString
    element_type: ElementType
    text: NonEmptyString
    heading_path: list[str] = Field(default_factory=list)
    page_number: int | None = Field(default=None, ge=1)
    order: int = Field(ge=0)
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_offsets(self) -> Self:
        """Require offsets to be paired and ordered when a parser supplies them."""
        if (self.char_start is None) != (self.char_end is None):
            raise ValueError("char_start and char_end must either both be set or both be absent")
        if (
            self.char_start is not None
            and self.char_end is not None
            and self.char_end <= self.char_start
        ):
            raise ValueError("char_end must be greater than char_start")
        return self


class NormalizedDocument(DomainModel):
    """Canonical parser output consumed by every chunking strategy."""

    document_id: NonEmptyString
    document_version: NonEmptyString
    source_name: NonEmptyString
    source_identifier: NonEmptyString
    mime_type: NonEmptyString
    title: str | None = None
    language: str | None = None
    text: NonEmptyString
    elements: list[DocumentElement] = Field(min_length=1)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    parser_name: NonEmptyString
    parser_version: NonEmptyString
    content_checksum: NonEmptyString

    @model_validator(mode="after")
    def validate_element_order(self) -> Self:
        """Reject duplicate or non-monotonic element positions."""
        orders = [element.order for element in self.elements]
        if orders != sorted(orders) or len(orders) != len(set(orders)):
            raise ValueError("document elements must have unique ascending order values")
        return self


class Chunk(DomainModel):
    """Traceable text unit embedded and stored as one vector point."""

    chunk_id: NonEmptyString
    document_id: NonEmptyString
    document_version: NonEmptyString
    parent_chunk_id: str | None = None
    chunk_index: int = Field(ge=0)
    content: NonEmptyString
    embedding_text: NonEmptyString
    search_text: str | None = None
    context_prefix: str | None = None
    heading_path: list[str] = Field(default_factory=list)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    token_count: int = Field(ge=1)
    content_type: NonEmptyString
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None
    content_checksum: NonEmptyString
    chunker_name: NonEmptyString
    chunker_version: NonEmptyString
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_range_and_links(self) -> Self:
        """Keep page ranges and chunk links internally consistent."""
        if self.page_start is not None and self.page_end is None:
            raise ValueError("page_end is required when page_start is set")
        if self.page_end is not None and self.page_start is None:
            raise ValueError("page_start is required when page_end is set")
        if (
            self.page_start is not None
            and self.page_end is not None
            and self.page_end < self.page_start
        ):
            raise ValueError("page_end must not be before page_start")
        linked_ids = (self.parent_chunk_id, self.previous_chunk_id, self.next_chunk_id)
        if self.chunk_id in linked_ids:
            raise ValueError("a chunk cannot link to itself")
        return self


class IngestionWarning(DomainModel):
    """Sanitized warning produced while processing one source document."""

    code: NonEmptyString
    message: NonEmptyString
    page_number: int | None = Field(default=None, ge=1)
    element_id: str | None = None


class IngestionResult(DomainModel):
    """Outcome for one document in a corpus ingestion run."""

    status: IngestionStatus
    warnings: list[IngestionWarning] = Field(default_factory=list)
    document_id: str | None = None
    document_version: str | None = None
    chunk_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        """Require identity for usable versions and no chunks for rejected sources."""
        if self.status in {"completed", "partial", "skipped"} and self.document_id is None:
            raise ValueError(f"document_id is required when status is {self.status}")
        if self.status in {"completed", "partial", "skipped"} and self.document_version is None:
            raise ValueError(f"document_version is required when status is {self.status}")
        if self.status == "rejected" and self.chunk_count != 0:
            raise ValueError("a rejected document cannot report stored chunks")
        return self


class RetrievalQuery(DomainModel):
    """Validated retrieval request independent of the vector-store client."""

    text: NonEmptyString
    top_k: int = Field(default=12, ge=1, le=100)
    document_ids: list[str] | None = None
    source_types: list[str] | None = None
    languages: list[str] | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None

    @model_validator(mode="after")
    def validate_date_range(self) -> Self:
        """Reject inverted metadata-filter ranges."""
        if self.date_from is not None and self.date_to is not None:
            if self.date_to < self.date_from:
                raise ValueError("date_to must not be before date_from")
        return self


class RetrievedChunk(DomainModel):
    """A ranked domain chunk returned by retrieval."""

    chunk: Chunk
    score: float = Field(allow_inf_nan=False)
    rank: int = Field(ge=1)


class ContextBudget(DomainModel):
    """Hard limits applied after retrieval and before generation."""

    max_tokens: int = Field(ge=1)
    max_chunks: int = Field(default=6, ge=1)


class AssembledContext(DomainModel):
    """Validated context selected and rendered for grounded generation."""

    query: NonEmptyString
    chunks: list[RetrievedChunk]
    rendered: str
    token_count: int = Field(ge=0)
    excluded_chunk_ids: list[str] = Field(default_factory=list)
