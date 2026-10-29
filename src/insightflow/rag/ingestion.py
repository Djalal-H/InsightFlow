"""Corpus ingestion orchestration and public ingestion-domain imports."""

from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path

from insightflow.core.exceptions import DocumentRejectedError, ProviderError
from insightflow.providers.embeddings import EmbeddingProvider
from insightflow.rag.chunkers import MaxMinSemanticChunker, StructureRecursiveChunker
from insightflow.rag.config import ChunkingConfig, EmbeddingConfig, IngestionConfig
from insightflow.rag.identity import canonicalize_source_identifier
from insightflow.rag.models import (
    Chunk,
    DocumentElement,
    DocumentSource,
    IngestionResult,
    IngestionStatus,
    IngestionWarning,
    NormalizedDocument,
)
from insightflow.rag.parsers import DoclingPdfParser
from insightflow.rag.parsers.docx import DocxParser
from insightflow.rag.parsers.markdown import MarkdownParser
from insightflow.rag.parsers.txt import TxtParser
from insightflow.rag.protocols import (
    ChunkContextualizer,
    Chunker,
    ChunkStore,
    DocumentParser,
)

UNKNOWN_MIME_TYPE = "application/octet-stream"
MIME_TYPES_BY_SUFFIX: Mapping[str, str] = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pdf": "application/pdf",
}


def discover_sources(corpus_dir: Path) -> list[DocumentSource]:
    """List every non-hidden file under an explicit corpus directory in a stable order."""
    if not corpus_dir.is_dir():
        raise ValueError("corpus_dir must be an existing directory")
    sources: list[DocumentSource] = []
    for path in corpus_dir.rglob("*"):
        relative = path.relative_to(corpus_dir)
        if not path.is_file() or any(part.startswith(".") for part in relative.parts):
            continue
        sources.append(
            DocumentSource(
                source_name=path.name,
                source_identifier=canonicalize_source_identifier(relative.as_posix()),
                path=path,
                mime_type=MIME_TYPES_BY_SUFFIX.get(path.suffix.lower(), UNKNOWN_MIME_TYPE),
            )
        )
    return sorted(sources, key=lambda source: source.source_identifier)


def create_default_parsers() -> dict[str, DocumentParser]:
    """Map each supported MIME type to its parser."""
    pdf_parser = DoclingPdfParser()
    return {
        "text/plain": TxtParser(),
        "text/markdown": MarkdownParser(),
        MIME_TYPES_BY_SUFFIX[".docx"]: DocxParser(),
        "application/pdf": pdf_parser,
        "application/x-pdf": pdf_parser,
    }


def create_chunker(
    config: ChunkingConfig,
    embedding_provider: EmbeddingProvider,
    *,
    embedding_batch_size: int = 32,
) -> Chunker:
    """Build the chunker that implements the configured strategy."""
    if config.strategy == "structure_recursive":
        return StructureRecursiveChunker()
    if config.strategy == "semantic_max_min":
        return MaxMinSemanticChunker(
            embedding_provider,
            embedding_batch_size=embedding_batch_size,
        )
    raise ValueError(f"Chunking strategy {config.strategy!r} is not implemented")


class IngestionPipeline:
    """Parse, chunk, embed, and store sources, skipping versions already indexed."""

    def __init__(
        self,
        *,
        parsers: Mapping[str, DocumentParser],
        chunker: Chunker,
        chunking_config: ChunkingConfig,
        embedding_provider: EmbeddingProvider,
        embedding_config: EmbeddingConfig,
        store: ChunkStore,
        ingestion_config: IngestionConfig | None = None,
    ) -> None:
        if chunking_config.strategy != chunker.strategy_name:
            raise ValueError("chunker does not implement the configured chunking strategy")
        self._parsers = dict(parsers)
        self._chunker = chunker
        self._chunking_config = chunking_config
        self._embedding_provider = embedding_provider
        self._embedding_config = embedding_config
        self._store = store
        self._ingestion_config = ingestion_config or IngestionConfig()

    async def ingest_corpus(self, corpus_dir: Path) -> list[IngestionResult]:
        """Prepare the collection, then ingest every source sequentially in stable order."""
        sources = discover_sources(corpus_dir)
        await self._store.ensure_collection(self._embedding_config)
        return [await self.ingest_document(source) for source in sources]

    async def ingest_document(self, source: DocumentSource) -> IngestionResult:
        """Ingest one source; the caller must have prepared the collection."""
        parser = self._parsers.get(source.mime_type)
        if parser is None:
            return _rejected(source, "unsupported_format")
        try:
            document = await parser.parse(source)
        except DocumentRejectedError as exc:
            return _rejected(source, exc.reason)

        fingerprint = create_index_fingerprint(
            document, self._chunker, self._chunking_config, self._embedding_config
        )
        if not self._ingestion_config.force and await self._store.has_document_version(
            document.document_id, document.document_version, fingerprint
        ):
            return _identified(source, document, "skipped", chunk_count=0)

        chunks = await self._chunker.chunk(document, self._chunking_config)
        if not chunks:
            return _rejected(source, "empty_document")
        _ensure_chunks_belong_to(document, chunks)
        vectors = await self._embed(chunks)
        await self._store.replace_document(document, chunks, vectors, fingerprint)
        return _identified(source, document, "completed", chunk_count=len(chunks))

    async def _embed(self, chunks: Sequence[Chunk]) -> list[list[float]]:
        """Embed chunk text in configured batches and validate every vector's size."""
        batch_size = self._ingestion_config.embedding_batch_size
        vectors: list[list[float]] = []
        for start in range(0, len(chunks), batch_size):
            texts = [chunk.embedding_text for chunk in chunks[start : start + batch_size]]
            batch = await self._embedding_provider.embed_documents(texts)
            if len(batch) != len(texts):
                raise ProviderError("The embedding provider returned an unexpected vector count")
            if any(len(vector) != self._embedding_config.dimensions for vector in batch):
                raise ProviderError(
                    "The embedding provider returned vectors of an unexpected size"
                )
            vectors.extend(batch)
        return vectors


def create_index_fingerprint(
    document: NormalizedDocument,
    chunker: Chunker,
    chunking_config: ChunkingConfig,
    embedding_config: EmbeddingConfig,
) -> str:
    """Identify every setting that changes a document's stored chunks or vectors."""
    parts = (
        f"parser:{document.parser_name}:{document.parser_version}",
        f"chunker:{chunker.strategy_name}:{chunker.strategy_version}",
        f"chunking:{chunking_config.model_dump_json()}",
        f"embedding:{embedding_config.fingerprint}",
    )
    return sha256("|".join(parts).encode("utf-8")).hexdigest()


def _identified(
    source: DocumentSource,
    document: NormalizedDocument,
    status: IngestionStatus,
    *,
    chunk_count: int,
) -> IngestionResult:
    return IngestionResult(
        source_identifier=source.source_identifier,
        status=status,
        document_id=document.document_id,
        document_version=document.document_version,
        chunk_count=chunk_count,
    )


def _rejected(source: DocumentSource, reason: str) -> IngestionResult:
    """Build a sanitized rejection outcome that never exposes file contents."""
    return IngestionResult(
        source_identifier=source.source_identifier,
        status="rejected",
        warnings=[IngestionWarning(code=reason, message=f"Document rejected: {reason}")],
    )


def _ensure_chunks_belong_to(document: NormalizedDocument, chunks: list[Chunk]) -> None:
    """Reject chunker output that is not traceable to the parsed document version."""
    for chunk in chunks:
        if (chunk.document_id, chunk.document_version) != (
            document.document_id,
            document.document_version,
        ):
            raise ValueError("chunker returned a chunk for a different document version")


__all__ = [
    "Chunk",
    "ChunkContextualizer",
    "ChunkStore",
    "Chunker",
    "DocumentElement",
    "DocumentParser",
    "DocumentSource",
    "IngestionPipeline",
    "IngestionResult",
    "IngestionWarning",
    "MaxMinSemanticChunker",
    "NormalizedDocument",
    "create_chunker",
    "create_index_fingerprint",
    "create_default_parsers",
    "discover_sources",
]
