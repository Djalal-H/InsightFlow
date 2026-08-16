"""Retrieval-augmented generation domain."""

from insightflow.rag.chunkers import MaxMinSemanticChunker
from insightflow.rag.config import (
    ChunkingConfig,
    EmbeddingConfig,
    IngestionConfig,
    RetrievalConfig,
)
from insightflow.rag.models import (
    AssembledContext,
    Chunk,
    ContextBudget,
    DocumentElement,
    DocumentSource,
    IngestionResult,
    IngestionWarning,
    NormalizedDocument,
    RetrievalQuery,
    RetrievedChunk,
)
from insightflow.rag.parsers import DoclingPdfParser
from insightflow.rag.protocols import (
    ChunkContextualizer,
    Chunker,
    ContextAssembler,
    DocumentParser,
    Retriever,
)

__all__ = [
    "AssembledContext",
    "Chunk",
    "ChunkContextualizer",
    "Chunker",
    "ChunkingConfig",
    "ContextAssembler",
    "ContextBudget",
    "DocumentElement",
    "DocumentParser",
    "DoclingPdfParser",
    "DocumentSource",
    "EmbeddingConfig",
    "IngestionConfig",
    "IngestionResult",
    "IngestionWarning",
    "MaxMinSemanticChunker",
    "NormalizedDocument",
    "RetrievalConfig",
    "RetrievalQuery",
    "RetrievedChunk",
    "Retriever",
]
