"""Tests for the Qdrant adapter's collection setup and payload mapping."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from qdrant_client import AsyncQdrantClient, models

from insightflow.core.config import Settings
from insightflow.core.exceptions import StorageError
from insightflow.rag.config import EmbeddingConfig
from insightflow.rag.models import Chunk, DocumentElement, NormalizedDocument
from insightflow.storage.qdrant import QdrantStore, chunk_to_payload, payload_to_chunk

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")
EMBEDDING = EmbeddingConfig(provider="fake", model="fake", dimensions=3, max_input_tokens=512)


@pytest.fixture
async def store() -> AsyncIterator[QdrantStore]:
    settings = Settings(_env_file=None, qdrant_collection="test_chunks")
    qdrant_store = QdrantStore(settings, client=AsyncQdrantClient(location=":memory:"))
    yield qdrant_store
    await qdrant_store.close()


def make_document() -> NormalizedDocument:
    return NormalizedDocument(
        document_id="11111111-1111-5111-8111-111111111111",
        document_version="v1",
        source_name="guide.pdf",
        source_identifier="PDF/guide.pdf",
        mime_type="application/pdf",
        title="Guide",
        text="Body.",
        elements=[
            DocumentElement(element_id="e1", element_type="paragraph", text="Body.", order=0)
        ],
        parser_name="docling_pdf",
        parser_version="1.0.0",
        content_checksum="checksum",
    )


def make_chunk(index: int) -> Chunk:
    return Chunk(
        chunk_id=f"22222222-2222-5222-8222-22222222222{index}",
        document_id="11111111-1111-5111-8111-111111111111",
        document_version="v1",
        chunk_index=index,
        content=f"Body {index}.",
        embedding_text=f"Body {index}.",
        heading_path=["Intro"],
        page_start=2,
        page_end=3,
        token_count=2,
        content_type="paragraph",
        content_checksum="checksum",
        chunker_name="structure_recursive",
        chunker_version="1.0.0",
        metadata={"element_ids": ["e1"]},
    )


def test_payload_round_trips_chunk_and_keeps_source_provenance() -> None:
    chunk = make_chunk(0)

    payload = chunk_to_payload(
        make_document(), chunk, index_fingerprint="fp", document_chunk_count=1
    )

    assert payload_to_chunk(payload) == chunk
    assert payload["source_identifier"] == "PDF/guide.pdf"
    assert payload["source_name"] == "guide.pdf"
    assert payload["page_start"] == 2
    assert payload["index_fingerprint"] == "fp"
    assert payload["document_chunk_count"] == 1


async def test_ensure_collection_creates_a_named_vector(store: QdrantStore) -> None:
    await store.ensure_collection(EMBEDDING)
    await store.ensure_collection(EMBEDDING)

    info = await store._client.get_collection(store.collection_name)
    vectors = info.config.params.vectors
    assert isinstance(vectors, dict)
    assert vectors["dense"].size == 3
    assert vectors["dense"].distance == models.Distance.COSINE


async def test_ensure_collection_refuses_mismatched_vectors(store: QdrantStore) -> None:
    await store.ensure_collection(EMBEDDING)

    with pytest.raises(StorageError, match="do not match"):
        await store.ensure_collection(EMBEDDING.model_copy(update={"dimensions": 4}))


async def test_document_version_is_complete_only_with_all_chunks(store: QdrantStore) -> None:
    await store.ensure_collection(EMBEDDING)
    document = make_document()
    chunks = [make_chunk(0), make_chunk(1)]
    doc_id = document.document_id

    assert not await store.has_document_version(doc_id, "v1", "fp")
    await store.replace_document(document, chunks, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], "fp")
    assert await store.has_document_version(doc_id, "v1", "fp")
    assert not await store.has_document_version(doc_id, "v1", "other-fp")

    await store._client.delete(
        store.collection_name,
        points_selector=models.PointIdsList(points=[chunks[1].chunk_id]),
    )
    assert not await store.has_document_version(doc_id, "v1", "fp")


async def test_replace_document_rejects_mismatched_lengths(store: QdrantStore) -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        await store.replace_document(make_document(), [make_chunk(0)], [], "fp")
