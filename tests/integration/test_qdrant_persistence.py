"""Opt-in persistence tests against a running Qdrant server.

Run with `QDRANT_TEST_URL=http://localhost:6333 pytest -m qdrant`. The tests use synthetic
documents and deterministic fake embeddings, so they make no hosted-model requests. Each test
uses its own collection and deletes it afterwards.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from uuid import uuid4

import pytest
from qdrant_client import AsyncQdrantClient

from insightflow.agents.memory import ConversationMemory
from insightflow.agents.models import ConversationTurn
from insightflow.core.config import Settings
from insightflow.rag.chunkers import StructureRecursiveChunker
from insightflow.rag.config import ChunkingConfig, EmbeddingConfig
from insightflow.rag.ingestion import IngestionPipeline
from insightflow.rag.models import RetrievalQuery
from insightflow.rag.parsers.txt import TxtParser
from insightflow.rag.retrieval import DenseRetriever, HybridRetriever
from insightflow.rag.sparse import Bm25SparseEncoder
from insightflow.storage.qdrant import QdrantStore
from insightflow.storage.qdrant_memory import QdrantMemoryStore
from tests.fakes import HYBRID_DOCUMENTS, HYBRID_QUERY

QDRANT_TEST_URL = os.environ.get("QDRANT_TEST_URL")

pytestmark = [
    pytest.mark.qdrant,
    pytest.mark.skipif(not QDRANT_TEST_URL, reason="QDRANT_TEST_URL is not set"),
]

KEYWORDS = ("river", "mountain", "desert")
EMBEDDING = EmbeddingConfig(
    provider="fake", model="keyword", dimensions=len(KEYWORDS), max_input_tokens=512
)


class KeywordEmbeddingProvider:
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(text.lower().count(word)) + 0.01 for word in KEYWORDS] for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]

    def count_tokens(self, text: str) -> int:
        return max(1, len(text.split()))


@pytest.fixture
async def settings() -> AsyncIterator[Settings]:
    test_settings = Settings(
        _env_file=None,
        qdrant_url=QDRANT_TEST_URL or "",
        qdrant_api_key=None,
        qdrant_collection=f"insightflow_test_{uuid4().hex}",
    )
    yield test_settings
    client = AsyncQdrantClient(url=test_settings.qdrant_url)
    await client.delete_collection(test_settings.qdrant_collection)
    await client.close()


def make_pipeline(store: QdrantStore, *, sparse: bool = False) -> IngestionPipeline:
    return IngestionPipeline(
        sparse_encoder=Bm25SparseEncoder() if sparse else None,
        parsers={"text/plain": TxtParser()},
        chunker=StructureRecursiveChunker(),
        chunking_config=ChunkingConfig(
            strategy="structure_recursive", target_tokens=100, max_tokens=200, min_tokens=1
        ),
        embedding_provider=KeywordEmbeddingProvider(),
        embedding_config=EMBEDDING,
        store=store,
    )


async def test_ingested_chunks_are_searchable_from_a_new_client(
    tmp_path: Path, settings: Settings
) -> None:
    (tmp_path / "river.txt").write_text("The river runs past the river bank.", encoding="utf-8")
    (tmp_path / "desert.txt").write_text("The desert is dry.", encoding="utf-8")
    writer = QdrantStore(settings)
    results = await make_pipeline(writer).ingest_corpus(tmp_path)
    await writer.close()
    assert [result.status for result in results] == ["completed", "completed"]

    reader = QdrantStore(settings)
    try:
        hits = await DenseRetriever(KeywordEmbeddingProvider(), reader).retrieve(
            RetrievalQuery(text="river", top_k=1)
        )
        second_run = await make_pipeline(reader).ingest_corpus(tmp_path)
        info = await reader._client.get_collection(settings.qdrant_collection)
    finally:
        await reader.close()

    assert [hit.source_identifier for hit in hits] == ["river.txt"]
    assert [result.status for result in second_run] == ["skipped", "skipped"]
    assert {
        "document_id",
        "document_version",
        "index_fingerprint",
        "source_identifier",
        "mime_type",
    } <= set(info.payload_schema)


async def test_hybrid_search_works_on_a_server_and_reads_chunks(
    tmp_path: Path, settings: Settings
) -> None:
    """The server applies IDF to stored sparse vectors and fuses both rankings."""
    for name, text in HYBRID_DOCUMENTS.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    store = QdrantStore(settings)
    try:
        results = await make_pipeline(store, sparse=True).ingest_corpus(tmp_path)
        query = RetrievalQuery(text=HYBRID_QUERY, top_k=1)
        dense = await DenseRetriever(KeywordEmbeddingProvider(), store).retrieve(query)
        retriever = HybridRetriever(KeywordEmbeddingProvider(), store, Bm25SparseEncoder())
        hybrid = await retriever.retrieve(query)
        filtered = await retriever.retrieve(
            RetrievalQuery(text=HYBRID_QUERY, source_identifiers=["cs101.txt"])
        )
        read = await store.get_chunks([hybrid[0].chunk.chunk_id])
        info = await store._client.get_collection(settings.qdrant_collection)
    finally:
        await store.close()

    assert [result.status for result in results] == ["completed"] * len(HYBRID_DOCUMENTS)
    assert [hit.source_identifier for hit in dense] == ["rivers.txt"]
    assert [hit.source_identifier for hit in hybrid] == ["cs303.txt"]
    assert [hit.source_identifier for hit in filtered] == ["cs101.txt"]
    assert read == [hybrid[0].chunk]
    assert "sparse" in (info.config.params.sparse_vectors or {})


async def test_conversation_memory_persists_on_a_server(settings: Settings) -> None:
    memory_settings = settings.model_copy(
        update={"memory_collection": f"{settings.qdrant_collection}_memory"}
    )
    turn = ConversationTurn(query="Where is the river?", answer="Past the mountain.")
    writer = QdrantMemoryStore(memory_settings)
    reader = QdrantMemoryStore(memory_settings)
    try:
        await ConversationMemory(KeywordEmbeddingProvider(), writer).remember("t1", 0, turn)
        memory = ConversationMemory(KeywordEmbeddingProvider(), reader)
        recalled = await memory.recall("t1", "river", before_turn=1)
        other = await memory.recall("t2", "river", before_turn=1)
        await memory.forget("t1")
        after = await memory.recall("t1", "river", before_turn=1)
    finally:
        await writer.close()
        await reader._client.delete_collection(memory_settings.memory_collection)
        await reader.close()

    assert (recalled, other, after) == ([turn], [], [])
