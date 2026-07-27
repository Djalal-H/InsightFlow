"""Small implementation boundaries for the Stage 3 RAG pipeline."""

from typing import Protocol, runtime_checkable

from insightflow.rag.config import ChunkingConfig
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

    def chunk(self, document: NormalizedDocument, config: ChunkingConfig) -> list[Chunk]: ...


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

