"""Retrieval-augmented generation domain.

`DoclingPdfParser` and `MaxMinSemanticChunker` load on first use, so the API process (which
needs neither) does not import docling, its model stack, or the chunker's provider code.
"""

from typing import TYPE_CHECKING, Any

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
from insightflow.rag.protocols import (
    ChunkContextualizer,
    Chunker,
    ContextAssembler,
    DocumentParser,
    Retriever,
)

if TYPE_CHECKING:
    from insightflow.rag.chunkers.max_min import MaxMinSemanticChunker
    from insightflow.rag.parsers.docling_pdf import DoclingPdfParser


def __getattr__(name: str) -> Any:
    """Import the heavy exports only when they are used (PEP 562)."""
    if name == "DoclingPdfParser":
        from insightflow.rag.parsers.docling_pdf import DoclingPdfParser

        return DoclingPdfParser
    if name == "MaxMinSemanticChunker":
        from insightflow.rag.chunkers.max_min import MaxMinSemanticChunker

        return MaxMinSemanticChunker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
