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

from insightflow.core.config import Settings
from insightflow.rag.chunkers import StructureRecursiveChunker
from insightflow.rag.config import ChunkingConfig, EmbeddingConfig
from insightflow.rag.ingestion import IngestionPipeline
from insightflow.rag.models import RetrievalQuery
from insightflow.rag.parsers.txt import TxtParser
from insightflow.rag.retrieval import DenseRetriever
from insightflow.storage.qdrant import QdrantStore

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


def make_pipeline(store: QdrantStore) -> IngestionPipeline:
    return IngestionPipeline(
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
