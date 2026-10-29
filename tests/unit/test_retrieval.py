"""Tests for dense retrieval and budgeted context assembly."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient

from insightflow.core.config import Settings
from insightflow.rag.chunkers import StructureRecursiveChunker
from insightflow.rag.config import ChunkingConfig, EmbeddingConfig
from insightflow.rag.ingestion import IngestionPipeline
from insightflow.rag.models import Chunk, ContextBudget, RetrievalQuery, RetrievedChunk
from insightflow.rag.parsers.markdown import MarkdownParser
from insightflow.rag.parsers.txt import TxtParser
from insightflow.rag.retrieval import (
    BudgetedContextAssembler,
    DenseRetriever,
    render_context_block,
)
from insightflow.storage.qdrant import QdrantStore

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")

KEYWORDS = ("river", "mountain", "desert")
EMBEDDING = EmbeddingConfig(
    provider="fake", model="keyword", dimensions=len(KEYWORDS), max_input_tokens=512
)


class KeywordEmbeddingProvider:
    """Embeds text as keyword counts so similarity is predictable."""

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(text.lower().count(word)) + 0.01 for word in KEYWORDS] for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]

    def count_tokens(self, text: str) -> int:
        return max(1, len(text.split()))


@pytest.fixture
async def store() -> AsyncIterator[QdrantStore]:
    settings = Settings(_env_file=None, qdrant_collection="test_retrieval")
    qdrant_store = QdrantStore(settings, client=AsyncQdrantClient(location=":memory:"))
    yield qdrant_store
    await qdrant_store.close()


async def ingest(store: QdrantStore, corpus: Path) -> None:
    (corpus / "river.txt").write_text("The river runs past the river bank.", encoding="utf-8")
    (corpus / "mountain.md").write_text("# Peaks\n\nThe mountain is tall.", encoding="utf-8")
    (corpus / "desert.txt").write_text("The desert is dry.", encoding="utf-8")
    pipeline = IngestionPipeline(
        parsers={"text/plain": TxtParser(), "text/markdown": MarkdownParser()},
        chunker=StructureRecursiveChunker(),
        chunking_config=ChunkingConfig(
            strategy="structure_recursive", target_tokens=100, max_tokens=200, min_tokens=1
        ),
        embedding_provider=KeywordEmbeddingProvider(),
        embedding_config=EMBEDDING,
        store=store,
    )
    await pipeline.ingest_corpus(corpus)


async def test_dense_retriever_ranks_stored_chunks_with_sources(
    tmp_path: Path, store: QdrantStore
) -> None:
    await ingest(store, tmp_path)
    retriever = DenseRetriever(KeywordEmbeddingProvider(), store)

    results = await retriever.retrieve(RetrievalQuery(text="Where is the river?", top_k=2))

    assert [result.rank for result in results] == [1, 2]
    assert results[0].source_identifier == "river.txt"
    assert results[0].source_name == "river.txt"
    assert results[0].chunk.content == "The river runs past the river bank."
    assert results[0].score >= results[1].score


async def test_dense_retriever_applies_filters(tmp_path: Path, store: QdrantStore) -> None:
    await ingest(store, tmp_path)
    retriever = DenseRetriever(KeywordEmbeddingProvider(), store)

    by_type = await retriever.retrieve(
        RetrievalQuery(text="river", source_types=["text/markdown"])
    )
    river_id = (await retriever.retrieve(RetrievalQuery(text="river", top_k=1)))[0]
    by_document = await retriever.retrieve(
        RetrievalQuery(text="desert", document_ids=[river_id.chunk.document_id])
    )

    assert [result.source_identifier for result in by_type] == ["mountain.md"]
    assert [result.source_identifier for result in by_document] == ["river.txt"]


async def test_dense_retriever_applies_the_score_threshold(
    tmp_path: Path, store: QdrantStore
) -> None:
    await ingest(store, tmp_path)
    retriever = DenseRetriever(KeywordEmbeddingProvider(), store, score_threshold=0.9)

    results = await retriever.retrieve(RetrievalQuery(text="river"))

    assert [result.source_identifier for result in results] == ["river.txt"]


async def test_dense_retriever_rejects_unsupported_filters(store: QdrantStore) -> None:
    retriever = DenseRetriever(KeywordEmbeddingProvider(), store)

    with pytest.raises(ValueError, match="not supported"):
        await retriever.retrieve(
            RetrievalQuery(text="river", date_from=datetime(2026, 1, 1, tzinfo=UTC))
        )


def make_candidate(rank: int, words: int, **chunk_fields: object) -> RetrievedChunk:
    content = " ".join(["word"] * words)
    chunk = Chunk.model_validate(
        {
            "chunk_id": f"chunk-{rank}",
            "document_id": "doc",
            "document_version": "v1",
            "chunk_index": rank,
            "content": content,
            "embedding_text": content,
            "token_count": words,
            "content_type": "paragraph",
            "content_checksum": "checksum",
            "chunker_name": "structure_recursive",
            "chunker_version": "1.0.0",
            **chunk_fields,
        }
    )
    return RetrievedChunk(
        chunk=chunk,
        source_identifier=f"docs/{rank}.txt",
        source_name=f"{rank}.txt",
        score=1.0 / rank,
        rank=rank,
    )


def count_words(text: str) -> int:
    return len(text.split())


def test_render_context_block_includes_pages_and_headings() -> None:
    single = make_candidate(1, 2, page_start=3, page_end=3)
    spanning = make_candidate(2, 2, page_start=3, page_end=5, heading_path=["A", "B"])

    assert render_context_block(1, single) == "[1] docs/1.txt (page 3)\nword word"
    assert render_context_block(2, spanning) == (
        "[2] docs/2.txt (pages 3-5)\nSection: A > B\nword word"
    )


def test_assembler_selects_by_rank_within_the_chunk_limit() -> None:
    candidates = [make_candidate(rank, 3) for rank in (3, 1, 2)]

    context = BudgetedContextAssembler(count_words).assemble(
        "question", candidates, ContextBudget(max_tokens=1000, max_chunks=2)
    )

    assert [item.rank for item in context.chunks] == [1, 2]
    assert context.excluded_chunk_ids == ["chunk-3"]
    assert context.rendered.startswith("[1] docs/1.txt\n")
    assert "\n\n[2] docs/2.txt\n" in context.rendered


def test_assembler_skips_passages_that_exceed_the_token_budget() -> None:
    candidates = [make_candidate(1, 4), make_candidate(2, 50), make_candidate(3, 4)]

    context = BudgetedContextAssembler(count_words).assemble(
        "question", candidates, ContextBudget(max_tokens=12, max_chunks=6)
    )

    assert [item.rank for item in context.chunks] == [1, 3]
    assert context.excluded_chunk_ids == ["chunk-2"]
    assert context.token_count == 12
    assert "[2] docs/3.txt" in context.rendered
