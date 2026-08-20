"""Tests for bounded Max-Min semantic chunking."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import cast

import pytest

from insightflow.core.exceptions import ProviderError
from insightflow.rag.chunkers import MaxMinSemanticChunker
from insightflow.rag.config import ChunkingConfig
from insightflow.rag.identity import (
    content_checksum,
    create_document_id,
    create_document_version,
)
from insightflow.rag.models import DocumentElement, ElementType, NormalizedDocument


class FakeEmbeddingProvider:
    """Return configured vectors and count whitespace-delimited tokens."""

    def __init__(self, vectors: Sequence[Sequence[float]]) -> None:
        self._vectors = [list(vector) for vector in vectors]
        self.calls: list[list[str]] = []
        self._offset = 0

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        values = list(texts)
        self.calls.append(values)
        start = self._offset
        self._offset += len(values)
        return self._vectors[start : self._offset]

    async def embed_query(self, text: str) -> list[float]:
        del text
        return self._vectors[0]

    def count_tokens(self, text: str) -> int:
        return len(re.findall(r"\S+", text))


def make_document(elements: Sequence[DocumentElement]) -> NormalizedDocument:
    """Build a normalized document from ordered test elements."""
    text = "\n\n".join(element.text for element in elements)
    return NormalizedDocument(
        document_id=create_document_id("guides/semantic.md"),
        document_version=create_document_version(text),
        source_name="semantic.md",
        source_identifier="guides/semantic.md",
        mime_type="text/markdown",
        title="Semantic guide",
        text=text,
        elements=list(elements),
        parser_name="markdown",
        parser_version="1",
        content_checksum=content_checksum(text),
    )


def make_element(
    text: str,
    *,
    order: int = 0,
    element_type: ElementType = "paragraph",
    heading_path: Sequence[str] = (),
    page_number: int | None = None,
) -> DocumentElement:
    """Build one source element with deterministic provenance."""
    return DocumentElement(
        element_id=f"element-{order}",
        element_type=element_type,
        text=text,
        heading_path=list(heading_path),
        page_number=page_number,
        order=order,
    )


def semantic_config(**updates: object) -> ChunkingConfig:
    """Create compact token limits for deterministic unit tests."""
    values: dict[str, object] = {
        "strategy": "semantic_max_min",
        "min_tokens": 1,
        "target_tokens": 20,
        "max_tokens": 40,
    }
    values.update(updates)
    return ChunkingConfig.model_validate(values)


@pytest.mark.asyncio
async def test_chunker_splits_english_sentences_and_batches_embeddings() -> None:
    """Abbreviations and decimals stay intact while semantic units are batched."""
    document = make_document(
        [make_element("Dr. Ada measured 3.14 units. The result held! A new topic followed?")]
    )
    provider = FakeEmbeddingProvider(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [-1.0, 0.0],
        ]
    )
    chunker = MaxMinSemanticChunker(provider, embedding_batch_size=2)

    chunks = await chunker.chunk(document, semantic_config())

    assert provider.calls == [
        ["Dr. Ada measured 3.14 units.", "The result held!"],
        ["A new topic followed?"],
    ]
    assert [chunk.content for chunk in chunks] == [
        "Dr. Ada measured 3.14 units. The result held!",
        "A new topic followed?",
    ]


@pytest.mark.asyncio
async def test_tables_and_code_remain_atomic_semantic_units() -> None:
    """Punctuation and newlines inside structured elements do not create sentences."""
    elements = [
        make_element("Metric | Value\nA. | 3.14", element_type="table", order=0),
        make_element("if ready:\n    run.step()", element_type="code", order=1),
    ]
    document = make_document(elements)
    provider = FakeEmbeddingProvider([[1.0, 0.0], [0.0, 1.0]])
    chunker = MaxMinSemanticChunker(provider)

    chunks = await chunker.chunk(document, semantic_config(hard_threshold=1.0))

    assert provider.calls == [[elements[0].text, elements[1].text]]
    assert [chunk.content_type for chunk in chunks] == ["table", "code"]
    assert [chunk.content for chunk in chunks] == [elements[0].text, elements[1].text]


@pytest.mark.asyncio
async def test_reference_threshold_comparison_is_strict() -> None:
    """A score equal to hard_threshold starts a new Max-Min cluster."""
    document = make_document([make_element("First sentence. Second sentence.")])
    provider = FakeEmbeddingProvider(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    chunker = MaxMinSemanticChunker(provider)

    chunks = await chunker.chunk(
        document,
        semantic_config(initialization_constant=1.0, hard_threshold=0.0),
    )

    assert [chunk.content for chunk in chunks] == ["First sentence.", "Second sentence."]


@pytest.mark.asyncio
async def test_oversized_semantic_cluster_is_split_with_hard_token_limit() -> None:
    """Related sentences remain bounded even when Max-Min groups all of them."""
    document = make_document(
        [make_element("Alpha one. Alpha two. Alpha three. Alpha four.")]
    )
    provider = FakeEmbeddingProvider([[1.0, 0.0]] * 4)
    chunker = MaxMinSemanticChunker(provider)

    chunks = await chunker.chunk(
        document,
        semantic_config(target_tokens=4, max_tokens=4),
    )

    assert [chunk.content for chunk in chunks] == [
        "Alpha one. Alpha two.",
        "Alpha three. Alpha four.",
    ]
    assert all(chunk.token_count <= 4 for chunk in chunks)


@pytest.mark.asyncio
async def test_undersized_chunk_merges_with_most_similar_legal_neighbor() -> None:
    """Best-effort minimum merging preserves the hard maximum."""
    document = make_document([make_element("Alpha. Beta. Gamma.")])
    provider = FakeEmbeddingProvider(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.1, 0.995],
        ]
    )
    chunker = MaxMinSemanticChunker(provider)

    chunks = await chunker.chunk(
        document,
        semantic_config(
            min_tokens=2,
            target_tokens=2,
            max_tokens=2,
            hard_threshold=0.9,
            initialization_constant=0.5,
        ),
    )

    assert [chunk.content for chunk in chunks] == ["Alpha.", "Beta. Gamma."]
    assert chunks[0].metadata["below_min_tokens"] is True
    assert chunks[1].metadata["below_min_tokens"] is False


@pytest.mark.asyncio
async def test_chunk_traceability_links_and_forced_splits_are_deterministic() -> None:
    """Chunk output retains element, heading, page, identity, and size provenance."""
    elements = [
        make_element(
            "Semantic Chunking",
            order=0,
            element_type="heading",
            heading_path=["Semantic Chunking"],
            page_number=1,
        ),
        make_element(
            "one two three four five six seven",
            order=1,
            heading_path=["Semantic Chunking"],
            page_number=2,
        ),
    ]
    document = make_document(elements)
    provider = FakeEmbeddingProvider([[1.0, 0.0]] * 4)
    chunker = MaxMinSemanticChunker(provider)
    config = semantic_config(target_tokens=2, max_tokens=3)

    chunks = await chunker.chunk(document, config)

    assert all(chunk.token_count <= 3 for chunk in chunks)
    assert any(chunk.metadata["forced_token_split"] is True for chunk in chunks)
    assert chunks[0].heading_path == ["Semantic Chunking"]
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 2
    assert chunks[0].next_chunk_id == chunks[1].chunk_id
    assert chunks[1].previous_chunk_id == chunks[0].chunk_id
    assert chunks[0].chunker_name == "semantic_max_min"
    assert chunks[0].embedding_text == chunks[0].content

    second_provider = FakeEmbeddingProvider([[1.0, 0.0]] * 4)
    repeated = await MaxMinSemanticChunker(second_provider).chunk(document, config)
    assert [chunk.chunk_id for chunk in repeated] == [chunk.chunk_id for chunk in chunks]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "vectors, message",
    [
        ([[1.0, 0.0]], "vector count"),
        ([[1.0, 0.0], [1.0]], "dimensions"),
        ([[1.0, 0.0], [0.0, 0.0]], "zero-length"),
        ([[1.0, 0.0], [math.nan, 1.0]], "invalid vector"),
        ([[1.0, 0.0], cast(list[float], ["invalid", 1.0])], "invalid vector"),
    ],
)
async def test_invalid_embedding_results_are_rejected(
    vectors: Sequence[Sequence[float]],
    message: str,
) -> None:
    """Malformed hosted embedding output fails before chunk construction."""
    document = make_document([make_element("First sentence. Second sentence.")])
    chunker = MaxMinSemanticChunker(FakeEmbeddingProvider(vectors))

    with pytest.raises(ProviderError, match=message):
        await chunker.chunk(document, semantic_config())
