"""Small implementation boundaries for the Stage 3 RAG pipeline."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from insightflow.rag.config import ChunkingConfig, EmbeddingConfig
from insightflow.rag.models import (
    AssembledContext,
    Chunk,
    ContextBudget,
    DocumentSource,
    NormalizedDocument,
    RetrievalQuery,
    RetrievedChunk,
)


@runtime_checkable
class DocumentParser(Protocol):
    """Convert one source into the shared normalized representation."""

    parser_name: str
    parser_version: str

    async def parse(self, source: DocumentSource) -> NormalizedDocument: ...


@runtime_checkable
class Chunker(Protocol):
    """Create deterministic chunks from a normalized document."""

    strategy_name: str
    strategy_version: str

    async def chunk(
        self,
        document: NormalizedDocument,
        config: ChunkingConfig,
    ) -> list[Chunk]: ...


@runtime_checkable
class ChunkContextualizer(Protocol):
    """Enrich embedding text without replacing original chunk content."""

    contextualizer_name: str
    contextualizer_version: str

    async def contextualize(
        self,
        document: NormalizedDocument,
        chunks: list[Chunk],
    ) -> list[Chunk]: ...


@runtime_checkable
class Retriever(Protocol):
    """Return ranked chunks without leaking vector-store response types."""

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]: ...


@runtime_checkable
class ContextAssembler(Protocol):
    """Select and render retrieved candidates under explicit hard limits."""

    def assemble(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        budget: ContextBudget,
    ) -> AssembledContext: ...


@runtime_checkable
class ChunkStore(Protocol):
    """Persist embedded chunks without exposing vector-store client types."""

    async def ensure_collection(self, embedding: EmbeddingConfig) -> None: ...

    async def has_document_version(
        self,
        document_id: str,
        document_version: str,
        index_fingerprint: str,
    ) -> bool: ...

    async def replace_document(
        self,
        document: NormalizedDocument,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        index_fingerprint: str,
    ) -> None: ...


@runtime_checkable
class ChunkSearcher(Protocol):
    """Rank stored chunks by vector similarity and return domain results."""

    async def search_chunks(
        self,
        vector: Sequence[float],
        *,
        limit: int,
        document_ids: Sequence[str] | None = None,
        mime_types: Sequence[str] | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]: ...
