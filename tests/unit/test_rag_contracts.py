"""Contract tests for Stage 3 provider-neutral RAG values."""

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from insightflow.core.config import Settings
from insightflow.rag.config import ChunkingConfig, EmbeddingConfig
from insightflow.rag.identity import (
    canonicalize_source_identifier,
    content_checksum,
    create_chunk_id,
    create_document_id,
    create_document_version,
    create_element_id,
)
from insightflow.rag.models import (
    Chunk,
    DocumentElement,
    DocumentSource,
    IngestionResult,
    NormalizedDocument,
    RetrievalQuery,
    RetrievedChunk,
)
from insightflow.rag.protocols import Chunker, DocumentParser, Retriever


def make_element(*, order: int = 0) -> DocumentElement:
    """Build one valid element for model-validation tests."""
    return DocumentElement(
        element_id=f"element-{order}",
        element_type="paragraph",
        text="InsightFlow uses traceable document chunks.",
        order=order,
        char_start=0,
        char_end=43,
    )


def make_document() -> NormalizedDocument:
    """Build one valid normalized document."""
    text = "InsightFlow uses traceable document chunks."
    return NormalizedDocument(
        document_id=create_document_id("guides/overview.md"),
        document_version=create_document_version(text),
        source_name="overview.md",
        source_identifier="guides/overview.md",
        mime_type="text/markdown",
        title="Overview",
        text=text,
        elements=[make_element()],
        parser_name="markdown",
        parser_version="1",
        content_checksum=content_checksum(text),
    )


def make_chunk() -> Chunk:
    """Build one valid traceable chunk."""
    document = make_document()
    chunk_id = create_chunk_id(
        document_id=document.document_id,
        document_version=document.document_version,
        chunker_name="fixed_window",
        chunker_version="1",
        chunk_index=0,
        content=document.text,
    )
    return Chunk(
        chunk_id=chunk_id,
        document_id=document.document_id,
        document_version=document.document_version,
        chunk_index=0,
        content=document.text,
        embedding_text=f"Document: {document.title}\n\n{document.text}",
        token_count=8,
        content_type="paragraph",
        content_checksum=content_checksum(document.text),
        chunker_name="fixed_window",
        chunker_version="1",
    )


def test_document_source_keeps_disk_location_separate_from_stable_identifier() -> None:
    """Moving a corpus root does not have to alter document identity."""
    source = DocumentSource(
        source_name="overview.md",
        source_identifier="guides/overview.md",
        path=Path("/different/corpus/guides/overview.md"),
        mime_type="text/markdown",
    )

    assert source.path.is_absolute()
    assert create_document_id(source.source_identifier) == create_document_id(
        "guides/overview.md"
    )


@pytest.mark.parametrize(
    "value",
    ["", ".", "/absolute/file.md", "../outside.md", "folder/../../outside.md"],
)
def test_canonical_source_identifier_rejects_unsafe_paths(value: str) -> None:
    """Document identity cannot escape or collapse to the corpus root."""
    with pytest.raises(ValueError):
        canonicalize_source_identifier(value)


def test_canonical_source_identifier_normalizes_separators_and_unicode() -> None:
    """Equivalent corpus-relative spellings produce the same identifier."""
    assert canonicalize_source_identifier("./guides\\café.md") == "guides/café.md"


def test_content_and_document_identities_are_stable_and_sensitive() -> None:
    """The same inputs reproduce IDs while relevant changes create new values."""
    source = "guides/overview.md"
    content = "Stable normalized content"
    document_id = create_document_id(source)
    version = create_document_version(content)
    element_id = create_element_id(document_id, "1", 0, content)
    chunk_id = create_chunk_id(
        document_id=document_id,
        document_version=version,
        chunker_name="fixed_window",
        chunker_version="1",
        chunk_index=0,
        content=content,
    )

    assert create_document_id(source) == document_id
    assert create_document_version(content) == version
    assert create_element_id(document_id, "1", 0, content) == element_id
    assert UUID(chunk_id).version == 5
    assert create_document_version(f"{content}.") != version
    assert (
        create_chunk_id(
            document_id=document_id,
            document_version=version,
            chunker_name="fixed_window",
            chunker_version="1",
            chunk_index=1,
            content=content,
        )
        != chunk_id
    )


def test_document_element_requires_paired_ordered_offsets() -> None:
    """Parsers cannot publish ambiguous source offsets."""
    with pytest.raises(ValidationError, match="both be set"):
        DocumentElement(
            element_id="element",
            element_type="paragraph",
            text="content",
            order=0,
            char_start=0,
        )

    with pytest.raises(ValidationError, match="greater than"):
        DocumentElement(
            element_id="element",
            element_type="paragraph",
            text="content",
            order=0,
            char_start=4,
            char_end=4,
        )


def test_normalized_document_requires_unique_ascending_elements() -> None:
    """Chunkers receive one deterministic element sequence."""
    document = make_document()

    with pytest.raises(ValidationError, match="unique ascending"):
        document.model_copy(
            update={"elements": [make_element(order=1), make_element(order=0)]}
        ).model_validate(
            document.model_copy(
                update={"elements": [make_element(order=1), make_element(order=0)]}
            ).model_dump()
        )


def test_chunk_rejects_invalid_page_ranges_and_self_links() -> None:
    """Stored chunks cannot contain impossible traceability links."""
    chunk = make_chunk()

    with pytest.raises(ValidationError, match="page_end is required"):
        Chunk.model_validate(chunk.model_copy(update={"page_start": 2}).model_dump())

    with pytest.raises(ValidationError, match="cannot link to itself"):
        Chunk.model_validate(
            chunk.model_copy(update={"next_chunk_id": chunk.chunk_id}).model_dump()
        )


def test_ingestion_result_enforces_status_semantics() -> None:
    """Usable versions need identity and rejected versions cannot report chunks."""
    with pytest.raises(ValidationError, match="document_id is required"):
        IngestionResult(status="completed", chunk_count=1)

    with pytest.raises(ValidationError, match="cannot report stored chunks"):
        IngestionResult(status="rejected", chunk_count=1)


def test_retrieval_query_validates_limits_and_date_range() -> None:
    """Invalid search limits and inverted filters fail before storage calls."""
    with pytest.raises(ValidationError):
        RetrievalQuery(text="question", top_k=0)

    with pytest.raises(ValidationError, match="date_to"):
        RetrievalQuery(
            text="question",
            date_from=datetime(2026, 7, 27, tzinfo=UTC),
            date_to=datetime(2026, 7, 26, tzinfo=UTC),
        )


def test_chunking_and_embedding_configs_expose_locked_defaults() -> None:
    """Stage 3 defaults are explicit while the control remains configurable."""
    structural = ChunkingConfig()
    control = ChunkingConfig(
        strategy="fixed_window",
        min_tokens=1,
        target_tokens=512,
        max_tokens=512,
        overlap_tokens=64,
    )
    embedding = EmbeddingConfig(
        provider="hosted-provider",
        model="embedding-model",
        dimensions=1536,
        max_input_tokens=8191,
    )

    assert structural.model_dump() == {
        "strategy": "structure_recursive",
        "target_tokens": 450,
        "max_tokens": 600,
        "min_tokens": 80,
        "overlap_tokens": 0,
        "hard_threshold": 0.6,
        "similarity_coefficient": 0.9,
        "initialization_constant": 1.5,
    }
    assert control.overlap_tokens == 64
    assert embedding.fingerprint == (
        "hosted-provider:embedding-model:1536:cosine:provider_default"
    )


def test_settings_reject_incoherent_rag_token_limits() -> None:
    """Environment settings fail before constructing a chunker."""
    with pytest.raises(ValidationError, match="minimum <= target <= maximum"):
        Settings(
            _env_file=None,
            rag_chunk_min_tokens=500,
            rag_chunk_target_tokens=400,
            rag_chunk_max_tokens=600,
        )


def test_semantic_chunking_rejects_overlap() -> None:
    """Max-Min boundaries cannot be combined with fixed-window overlap."""
    with pytest.raises(ValidationError, match="does not support token overlap"):
        ChunkingConfig(strategy="semantic_max_min", overlap_tokens=1)

    with pytest.raises(ValidationError, match="does not support token overlap"):
        Settings(
            _env_file=None,
            rag_chunking_strategy="semantic_max_min",
            rag_chunk_overlap_tokens=1,
        )


def test_domain_models_are_strict_and_immutable() -> None:
    """Boundary values reject unknown fields and assignment."""
    document = make_document()

    with pytest.raises(ValidationError):
        NormalizedDocument.model_validate(
            {**document.model_dump(), "provider_specific_value": "forbidden"}
        )

    with pytest.raises(ValidationError):
        document.title = "Changed"  # type: ignore[misc]


def test_runtime_protocols_accept_provider_neutral_implementations() -> None:
    """Strategies can be selected without importing parser or storage SDK types."""
    class FakeParser:
        parser_name = "fake"
        parser_version = "1"

        async def parse(self, source: DocumentSource) -> NormalizedDocument:
            del source
            return make_document()

    class FakeChunker:
        strategy_name = "fake"
        strategy_version = "1"

        async def chunk(
            self,
            document: NormalizedDocument,
            config: ChunkingConfig,
        ) -> list[Chunk]:
            del document, config
            return [make_chunk()]

    class FakeRetriever:
        async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
            del query
            return [RetrievedChunk(chunk=make_chunk(), score=0.9, rank=1)]

    assert isinstance(FakeParser(), DocumentParser)
    assert isinstance(FakeChunker(), Chunker)
    assert isinstance(FakeRetriever(), Retriever)
