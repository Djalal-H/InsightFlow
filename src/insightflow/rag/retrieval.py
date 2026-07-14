"""Retrieval domain types and contracts."""

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """A retrieved passage and its relevance metadata."""

    text: str
    source: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class Retriever(Protocol):
    """Contract consumed by agent retrieval nodes."""

    async def retrieve(self, query: str, limit: int = 5) -> list[RetrievalResult]: ...

