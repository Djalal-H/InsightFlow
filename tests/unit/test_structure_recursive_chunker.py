"""Tests for the structure-aware (Track B) chunker."""

from __future__ import annotations

from insightflow.rag.chunkers.structure_recursive import (
    StructureRecursiveChunker,
    _count_tokens,
)
from insightflow.rag.config import ChunkingConfig
from insightflow.rag.models import DocumentElement, NormalizedDocument

MIME_DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def make_document(elements: list[DocumentElement]) -> NormalizedDocument:
    """Build a minimal NormalizedDocument fixture; chunking never reads `.text`."""
    return NormalizedDocument(
        document_id="doc-1",
        document_version="v1",
        source_name="sample.docx",
        source_identifier="DOCX/sample.docx",
        mime_type=MIME_DOCX,
        text="irrelevant for chunking",
        elements=elements,
        parser_name="control_docx_flatten",
        parser_version="1.0.0",
        content_checksum="irrelevant",
    )


def element(
    element_id: str,
    text: str,
    heading_path: list[str],
    order: int,
    element_type: str = "paragraph",
) -> DocumentElement:
    return DocumentElement(
        element_id=element_id,
        element_type=element_type,
        text=text,
        heading_path=heading_path,
        order=order,
    )


# ---- Section grouping -----------------------------------------------------


def test_chunker_produces_one_chunk_per_section_when_within_target() -> None:
    """Two distinct sections, each small, produce exactly two chunks."""
    elements = [
        element("e1", "Intro text.", ["Introduction"], 0),
        element("e2", "History text.", ["History"], 1),
    ]
    document = make_document(elements)
    config = ChunkingConfig(
        strategy="structure_recursive", target_tokens=100, max_tokens=200, min_tokens=1
    )

    chunks = StructureRecursiveChunker().chunk(document, config)

    assert len(chunks) == 2
    assert chunks[0].heading_path == ["Introduction"]
    assert chunks[1].heading_path == ["History"]


def test_chunker_splits_a_section_that_exceeds_target_tokens() -> None:
    """A section larger than target_tokens is split into multiple chunks."""
    long_text = "The university offers many academic programs. " * 10
    elements = [
        element("e1", long_text, ["Programs"], 0),
        element("e2", long_text, ["Programs"], 1),
        element("e3", long_text, ["Programs"], 2),
    ]
    document = make_document(elements)
    config = ChunkingConfig(
        strategy="structure_recursive", target_tokens=30, max_tokens=50, min_tokens=1
    )

    chunks = StructureRecursiveChunker().chunk(document, config)

    assert len(chunks) > 1
    assert all(chunk.heading_path == ["Programs"] for chunk in chunks)
    assert all(chunk.token_count <= config.max_tokens for chunk in chunks)


# ---- Oversized single elements --------------------------------------------


def test_chunker_splits_a_single_oversized_element_by_sentence() -> None:
    """One element bigger than max_tokens is broken into sentence-bounded pieces."""
    huge_text = "This is one sentence. " * 40
    elements = [element("e1", huge_text, ["Notes"], 0)]
    document = make_document(elements)
    config = ChunkingConfig(
        strategy="structure_recursive", target_tokens=20, max_tokens=30, min_tokens=1
    )

    chunks = StructureRecursiveChunker().chunk(document, config)

    assert len(chunks) > 1
    assert all(chunk.token_count <= config.max_tokens for chunk in chunks)
    rejoined = " ".join(chunk.content for chunk in chunks)
    assert "This is one sentence." in rejoined


# ---- Merging undersized chunks --------------------------------------------


def test_chunker_merges_undersized_trailing_chunk_within_same_section() -> None:
    """A tiny leftover chunk in the same section as its predecessor gets merged."""
    filler = "Word. " * 15  # deliberately crosses target_tokens once
    elements = [
        element("e1", filler, ["Section"], 0),
        element("e2", "Tiny.", ["Section"], 1),
    ]
    document = make_document(elements)
    config = ChunkingConfig(
        strategy="structure_recursive", target_tokens=8, max_tokens=100, min_tokens=5
    )

    chunks = StructureRecursiveChunker().chunk(document, config)

    # merged back into one chunk since both share the section and fit under max_tokens
    assert len(chunks) == 1
    assert "Tiny." in chunks[0].content


def test_chunker_leaves_trailing_undersized_section_alone_with_no_neighbor() -> None:
    """A short section with no same-heading neighbor cannot merge and stays on its own."""
    elements = [
        element("e1", "A reasonably sized introduction paragraph here.", ["Introduction"], 0),
        element("e2", "Tiny.", ["History"], 1),
    ]
    document = make_document(elements)
    config = ChunkingConfig(
        strategy="structure_recursive", target_tokens=50, max_tokens=100, min_tokens=20
    )

    chunks = StructureRecursiveChunker().chunk(document, config)

    assert len(chunks) == 2
    assert chunks[1].heading_path == ["History"]
    assert chunks[1].token_count < config.min_tokens  # documented, unavoidable edge case


# ---- Determinism and linking ----------------------------------------------


def test_chunker_output_is_deterministic() -> None:
    """The same document and config produce identical chunk IDs every time."""
    elements = [
        element("e1", "Intro text.", ["Introduction"], 0),
        element("e2", "History text.", ["History"], 1),
    ]
    document = make_document(elements)
    config = ChunkingConfig(strategy="structure_recursive")

    chunker = StructureRecursiveChunker()
    first = chunker.chunk(document, config)
    second = chunker.chunk(document, config)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]


def test_chunker_links_chunks_sequentially() -> None:
    """previous_chunk_id and next_chunk_id correctly chain across three chunks."""
    long_text = "The university offers many academic programs. " * 10
    elements = [
        element("e1", long_text, ["Programs"], 0),
        element("e2", long_text, ["Programs"], 1),
        element("e3", long_text, ["Programs"], 2),
    ]
    document = make_document(elements)
    config = ChunkingConfig(
        strategy="structure_recursive", target_tokens=30, max_tokens=50, min_tokens=1
    )

    chunks = StructureRecursiveChunker().chunk(document, config)
    assert len(chunks) >= 3

    assert chunks[0].previous_chunk_id is None
    assert chunks[-1].next_chunk_id is None
    for i in range(len(chunks) - 1):
        assert chunks[i].next_chunk_id == chunks[i + 1].chunk_id
        assert chunks[i + 1].previous_chunk_id == chunks[i].chunk_id


def test_chunker_never_exceeds_max_tokens() -> None:
    """No produced chunk exceeds the configured hard token ceiling."""
    long_text = "Sentence with several words in it. " * 20
    elements = [element("e1", long_text, ["Notes"], 0)]
    document = make_document(elements)
    config = ChunkingConfig(
        strategy="structure_recursive", target_tokens=15, max_tokens=25, min_tokens=1
    )

    chunks = StructureRecursiveChunker().chunk(document, config)

    for chunk in chunks:
        assert _count_tokens(chunk.content) <= config.max_tokens