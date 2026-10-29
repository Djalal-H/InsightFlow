"""Tests for corpus ingestion orchestration with fake embeddings and in-memory Qdrant."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient

from insightflow.core.config import Settings
from insightflow.core.exceptions import DocumentRejectedError, ProviderError
from insightflow.rag.chunkers import MaxMinSemanticChunker, StructureRecursiveChunker
from insightflow.rag.config import ChunkingConfig, EmbeddingConfig, IngestionConfig
from insightflow.rag.ingestion import (
    IngestionPipeline,
    create_chunker,
    create_index_fingerprint,
    discover_sources,
)
from insightflow.rag.models import Chunk, DocumentSource, NormalizedDocument
from insightflow.rag.parsers.markdown import MarkdownParser
from insightflow.rag.parsers.txt import TxtParser
from insightflow.rag.protocols import Chunker, DocumentParser
from insightflow.storage.qdrant import QdrantStore, payload_to_chunk

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")
STRUCTURE_CONFIG = ChunkingConfig(
    strategy="structure_recursive", target_tokens=100, max_tokens=200, min_tokens=1
)
EMBEDDING_CONFIG = EmbeddingConfig(
    provider="fake", model="fake-embedding", dimensions=3, max_input_tokens=512
)


class FakeEmbeddingProvider:
    """Returns deterministic vectors and records every batch it receives."""

    def __init__(self, dimensions: int = 3) -> None:
        self.dimensions = dimensions
        self.batches: list[list[str]] = []

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return [[float(len(text)), 1.0, 0.5][: self.dimensions] for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]

    def count_tokens(self, text: str) -> int:
        return max(1, len(text.split()))


class RejectingParser:
    parser_name = "rejecting"
    parser_version = "1.0.0"

    async def parse(self, source: DocumentSource) -> NormalizedDocument:
        raise DocumentRejectedError(
            reason="scanned_pdf", source_identifier=source.source_identifier
        )


class EmptyChunker:
    strategy_name = "structure_recursive"
    strategy_version = "1.0.0"

    async def chunk(self, document: NormalizedDocument, config: ChunkingConfig) -> list[Chunk]:
        return []


class ForeignChunker:
    """Returns chunks attributed to another document version."""

    strategy_name = "structure_recursive"
    strategy_version = "1.0.0"

    async def chunk(self, document: NormalizedDocument, config: ChunkingConfig) -> list[Chunk]:
        chunks = await StructureRecursiveChunker().chunk(document, config)
        return [chunk.model_copy(update={"document_version": "other"}) for chunk in chunks]


@pytest.fixture
async def store() -> AsyncIterator[QdrantStore]:
    settings = Settings(_env_file=None, qdrant_collection="test_chunks", rag_upload_batch_size=2)
    qdrant_store = QdrantStore(settings, client=AsyncQdrantClient(location=":memory:"))
    yield qdrant_store
    await qdrant_store.close()


def write_corpus(root: Path) -> None:
    (root / "notes").mkdir()
    (root / "notes" / "b.txt").write_text("Second document body.", encoding="utf-8")
    (root / "a.md").write_text("# Title\n\nFirst document body.", encoding="utf-8")
    (root / "image.png").write_bytes(b"\x89PNG")
    (root / ".hidden").mkdir()
    (root / ".hidden" / "secret.txt").write_text("ignored", encoding="utf-8")


def make_pipeline(
    store: QdrantStore,
    *,
    parsers: Mapping[str, DocumentParser] | None = None,
    chunker: Chunker | None = None,
    embedding_provider: FakeEmbeddingProvider | None = None,
    force: bool = False,
) -> IngestionPipeline:
    return IngestionPipeline(
        parsers=parsers or {"text/plain": TxtParser(), "text/markdown": MarkdownParser()},
        chunker=chunker or StructureRecursiveChunker(),
        chunking_config=STRUCTURE_CONFIG,
        embedding_provider=embedding_provider or FakeEmbeddingProvider(),
        embedding_config=EMBEDDING_CONFIG,
        store=store,
        ingestion_config=IngestionConfig(embedding_batch_size=1, force=force),
    )


async def stored_chunks(store: QdrantStore) -> list[Chunk]:
    points, _ = await store._client.scroll(store.collection_name, limit=100)
    return sorted(
        (payload_to_chunk(point.payload or {}) for point in points),
        key=lambda chunk: chunk.chunk_id,
    )


def test_discover_sources_is_sorted_relative_and_skips_hidden(tmp_path: Path) -> None:
    write_corpus(tmp_path)

    sources = discover_sources(tmp_path)

    assert [source.source_identifier for source in sources] == [
        "a.md",
        "image.png",
        "notes/b.txt",
    ]
    assert [source.mime_type for source in sources] == [
        "text/markdown",
        "application/octet-stream",
        "text/plain",
    ]


def test_discover_sources_requires_a_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="existing directory"):
        discover_sources(tmp_path / "missing")


async def test_ingest_corpus_embeds_and_stores_chunks(
    tmp_path: Path, store: QdrantStore
) -> None:
    write_corpus(tmp_path)
    embedder = FakeEmbeddingProvider()

    results = await make_pipeline(store, embedding_provider=embedder).ingest_corpus(tmp_path)

    statuses = {result.source_identifier: result.status for result in results}
    assert statuses == {"a.md": "completed", "image.png": "rejected", "notes/b.txt": "completed"}
    rejected = next(result for result in results if result.status == "rejected")
    assert [warning.code for warning in rejected.warnings] == ["unsupported_format"]
    stored = await stored_chunks(store)
    assert len(stored) == sum(result.chunk_count for result in results)
    assert {chunk.document_id for chunk in stored} == {
        result.document_id for result in results if result.status == "completed"
    }
    assert all(len(batch) == 1 for batch in embedder.batches)


async def test_unchanged_documents_are_skipped_without_embedding(
    tmp_path: Path, store: QdrantStore
) -> None:
    write_corpus(tmp_path)
    await make_pipeline(store).ingest_corpus(tmp_path)
    embedder = FakeEmbeddingProvider()

    results = await make_pipeline(store, embedding_provider=embedder).ingest_corpus(tmp_path)

    assert [result.status for result in results] == ["skipped", "rejected", "skipped"]
    assert all(result.chunk_count == 0 for result in results)
    assert embedder.batches == []


async def test_force_reingests_unchanged_documents(tmp_path: Path, store: QdrantStore) -> None:
    write_corpus(tmp_path)
    await make_pipeline(store).ingest_corpus(tmp_path)
    before = await stored_chunks(store)

    results = await make_pipeline(store, force=True).ingest_corpus(tmp_path)

    assert [result.status for result in results] == ["completed", "rejected", "completed"]
    assert await stored_chunks(store) == before


async def test_changed_document_replaces_its_previous_version(
    tmp_path: Path, store: QdrantStore
) -> None:
    source = tmp_path / "a.txt"
    source.write_text("Original body.", encoding="utf-8")
    await make_pipeline(store).ingest_corpus(tmp_path)
    source.write_text("Revised body.", encoding="utf-8")

    [result] = await make_pipeline(store).ingest_corpus(tmp_path)

    assert result.status == "completed"
    stored = await stored_chunks(store)
    assert {chunk.document_version for chunk in stored} == {result.document_version}
    assert [chunk.content for chunk in stored] == ["Revised body."]


async def test_parser_rejection_becomes_a_sanitized_result(
    tmp_path: Path, store: QdrantStore
) -> None:
    (tmp_path / "scan.txt").write_text("content", encoding="utf-8")

    [result] = await make_pipeline(
        store, parsers={"text/plain": RejectingParser()}
    ).ingest_corpus(tmp_path)

    assert result.status == "rejected"
    assert result.document_id is None
    assert [warning.code for warning in result.warnings] == ["scanned_pdf"]
    assert "content" not in result.warnings[0].message


async def test_document_without_chunks_is_rejected(tmp_path: Path, store: QdrantStore) -> None:
    (tmp_path / "a.txt").write_text("content", encoding="utf-8")

    [result] = await make_pipeline(store, chunker=EmptyChunker()).ingest_corpus(tmp_path)

    assert result.status == "rejected"
    assert result.warnings[0].code == "empty_document"


async def test_chunks_for_another_document_version_are_refused(
    tmp_path: Path, store: QdrantStore
) -> None:
    (tmp_path / "a.txt").write_text("content", encoding="utf-8")

    with pytest.raises(ValueError, match="different document version"):
        await make_pipeline(store, chunker=ForeignChunker()).ingest_corpus(tmp_path)


async def test_wrong_vector_size_stops_ingestion(tmp_path: Path, store: QdrantStore) -> None:
    (tmp_path / "a.txt").write_text("content", encoding="utf-8")
    pipeline = make_pipeline(store, embedding_provider=FakeEmbeddingProvider(dimensions=2))

    with pytest.raises(ProviderError, match="unexpected size"):
        await pipeline.ingest_corpus(tmp_path)
    assert await stored_chunks(store) == []


async def test_index_fingerprint_changes_with_chunking_settings(tmp_path: Path) -> None:
    source = tmp_path / "a.txt"
    source.write_text("content", encoding="utf-8")
    document = await TxtParser().parse(discover_sources(tmp_path)[0])
    chunker = StructureRecursiveChunker()
    wider = STRUCTURE_CONFIG.model_copy(update={"max_tokens": 300})

    first = create_index_fingerprint(document, chunker, STRUCTURE_CONFIG, EMBEDDING_CONFIG)
    second = create_index_fingerprint(document, chunker, wider, EMBEDDING_CONFIG)

    assert first != second
    assert first == create_index_fingerprint(
        document, chunker, STRUCTURE_CONFIG, EMBEDDING_CONFIG
    )


def test_pipeline_rejects_a_chunker_for_another_strategy(store: QdrantStore) -> None:
    with pytest.raises(ValueError, match="configured chunking strategy"):
        make_pipeline(store, chunker=MaxMinSemanticChunker(FakeEmbeddingProvider()))


def test_create_chunker_selects_the_configured_strategy() -> None:
    provider = FakeEmbeddingProvider()

    structure = create_chunker(STRUCTURE_CONFIG, provider)
    semantic = create_chunker(ChunkingConfig(strategy="semantic_max_min"), provider)

    assert isinstance(structure, StructureRecursiveChunker)
    assert isinstance(semantic, MaxMinSemanticChunker)
    with pytest.raises(ValueError, match="not implemented"):
        create_chunker(ChunkingConfig(strategy="fixed_window"), provider)
