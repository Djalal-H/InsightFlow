"""Document ingestion contracts; format-specific parsers are intentionally deferred."""

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """Normalized unit stored as one Qdrant point."""

    id: str
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentChunker(Protocol):
    """Boundary for a future format-aware chunking implementation."""

    def chunk(self, content: str, source: str) -> list[DocumentChunk]: ...

