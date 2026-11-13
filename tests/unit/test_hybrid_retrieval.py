"""Tests for hybrid search, source filters, and chunk reads on in-memory Qdrant."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient

from insightflow.core.config import Settings
from insightflow.core.exceptions import StorageError
from insightflow.rag.chunkers import StructureRecursiveChunker
from insightflow.rag.config import ChunkingConfig, EmbeddingConfig
from insightflow.rag.ingestion import (
    IngestionPipeline,
    create_index_fingerprint,
    discover_sources,
)
from insightflow.rag.models import RetrievalQuery
from insightflow.rag.parsers.txt import TxtParser
from insightflow.rag.retrieval import DenseRetriever, HybridRetriever
from insightflow.rag.sparse import Bm25SparseEncoder
from insightflow.storage.qdrant import QdrantStore
from tests.fakes import HYBRID_DOCUMENTS, HYBRID_QUERY, KEYWORDS, KeywordEmbeddingProvider

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")

EMBEDDING = EmbeddingConfig(
    provider="fake", model="keyword", dimensions=len(KEYWORDS), max_input_tokens=512
)
CHUNKING = ChunkingConfig(
    strategy="structure_recursive", target_tokens=100, max_tokens=200, min_tokens=1
)
DOCUMENTS = HYBRID_DOCUMENTS


@pytest.fixture
async def store() -> AsyncIterator[QdrantStore]:
    settings = Settings(_env_file=None, qdrant_collection="test_hybrid")
    qdrant_store = QdrantStore(settings, client=AsyncQdrantClient(location=":memory:"))
    yield qdrant_store
    await qdrant_store.close()


def pipeline(store: QdrantStore, *, sparse: bool) -> IngestionPipeline:
    return IngestionPipeline(
        parsers={"text/plain": TxtParser()},
        chunker=StructureRecursiveChunker(),
        chunking_config=CHUNKING,
        embedding_provider=KeywordEmbeddingProvider(),
        embedding_config=EMBEDDING,
        store=store,
        sparse_encoder=Bm25SparseEncoder() if sparse else None,
    )


async def ingest(store: QdrantStore, corpus: Path, *, sparse: bool = True) -> None:
    for name, text in DOCUMENTS.items():
        (corpus / name).write_text(text, encoding="utf-8")
    await pipeline(store, sparse=sparse).ingest_corpus(corpus)


async def test_hybrid_search_finds_an_exact_term_that_dense_search_ranks_second(
    tmp_path: Path, store: QdrantStore
) -> None:
    await ingest(store, tmp_path)
    provider = KeywordEmbeddingProvider()
    query = RetrievalQuery(text=HYBRID_QUERY, top_k=1)

    dense = await DenseRetriever(provider, store).retrieve(query)
    hybrid = await HybridRetriever(provider, store, Bm25SparseEncoder()).retrieve(query)

    assert [result.source_identifier for result in dense] == ["rivers.txt"]
    assert [result.source_identifier for result in hybrid] == ["cs303.txt"]
    assert hybrid[0].rank == 1


async def test_hybrid_search_applies_filters_to_both_rankings(
    tmp_path: Path, store: QdrantStore
) -> None:
    await ingest(store, tmp_path)
    retriever = HybridRetriever(KeywordEmbeddingProvider(), store, Bm25SparseEncoder())

    results = await retriever.retrieve(
        RetrievalQuery(text=HYBRID_QUERY, source_identifiers=["cs101.txt", "peak.txt"])
    )

    assert {result.source_identifier for result in results} == {"cs101.txt", "peak.txt"}


async def test_hybrid_search_without_query_terms_uses_dense_search(
    tmp_path: Path, store: QdrantStore
) -> None:
    await ingest(store, tmp_path)
    retriever = HybridRetriever(KeywordEmbeddingProvider(), store, Bm25SparseEncoder())

    results = await retriever.retrieve(RetrievalQuery(text="?!", top_k=4))

    assert len(results) == 4
    assert results[0].score <= 1.0  # a similarity, not a fusion score


async def test_dense_retriever_filters_by_source_identifier(
    tmp_path: Path, store: QdrantStore
) -> None:
    await ingest(store, tmp_path, sparse=False)
    retriever = DenseRetriever(KeywordEmbeddingProvider(), store)

    results = await retriever.retrieve(
        RetrievalQuery(text="river", source_identifiers=["peak.txt"])
    )

    assert [result.source_identifier for result in results] == ["peak.txt"]


async def test_hybrid_search_on_a_dense_only_collection_is_a_storage_error(
    tmp_path: Path, store: QdrantStore
) -> None:
    await ingest(store, tmp_path, sparse=False)
    retriever = HybridRetriever(KeywordEmbeddingProvider(), store, Bm25SparseEncoder())

    with pytest.raises(StorageError, match="search failed"):
        await retriever.retrieve(RetrievalQuery(text=HYBRID_QUERY))


async def test_sparse_ingestion_refuses_a_dense_only_collection(
    tmp_path: Path, store: QdrantStore
) -> None:
    await ingest(store, tmp_path, sparse=False)

    with pytest.raises(StorageError, match="no sparse vectors"):
        await pipeline(store, sparse=True).ingest_corpus(tmp_path)


async def test_sparse_encoder_changes_the_index_fingerprint_only_when_used(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.txt").write_text("Body.", encoding="utf-8")
    source = discover_sources(tmp_path)[0]
    document = await TxtParser().parse(source)
    chunker = StructureRecursiveChunker()

    dense = create_index_fingerprint(document, chunker, CHUNKING, EMBEDDING)
    same = create_index_fingerprint(document, chunker, CHUNKING, EMBEDDING, sparse_encoder=None)
    sparse = create_index_fingerprint(
        document, chunker, CHUNKING, EMBEDDING, sparse_encoder=Bm25SparseEncoder()
    )

    assert dense == same
    assert sparse != dense


async def test_unchanged_sparse_documents_are_skipped(tmp_path: Path) -> None:
    for name, text in DOCUMENTS.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    settings = Settings(_env_file=None, qdrant_collection="test_hybrid_reingest")
    store = QdrantStore(settings, client=AsyncQdrantClient(location=":memory:"))
    try:
        first = await pipeline(store, sparse=True).ingest_corpus(tmp_path)
        second = await pipeline(store, sparse=True).ingest_corpus(tmp_path)
    finally:
        await store.close()

    assert {result.status for result in first} == {"completed"}
    assert {result.status for result in second} == {"skipped"}


async def test_get_chunks_returns_stored_chunks_and_skips_unknown_ids(
    tmp_path: Path, store: QdrantStore
) -> None:
    await ingest(store, tmp_path)
    hit = (
        await DenseRetriever(KeywordEmbeddingProvider(), store).retrieve(
            RetrievalQuery(text="mountain", top_k=1)
        )
    )[0]

    chunks = await store.get_chunks(
        [hit.chunk.chunk_id, "00000000-0000-5000-8000-000000000000"]
    )

    assert chunks == [hit.chunk]
    assert await store.get_chunks([]) == []

