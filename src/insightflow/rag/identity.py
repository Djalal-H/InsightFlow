"""Deterministic identity and checksum helpers for ingestion."""

from hashlib import sha256
from pathlib import PurePosixPath
from unicodedata import normalize
from uuid import UUID, uuid5

INSIGHTFLOW_RAG_NAMESPACE = UUID("c77233cc-8939-4b9d-b882-5be7673cf89f")


def canonicalize_source_identifier(source_identifier: str) -> str:
    """Normalize a corpus-relative path without depending on the corpus root."""
    value = normalize("NFC", source_identifier.strip().replace("\\", "/"))
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError("source_identifier must be a non-empty corpus-relative path")
    normalized = path.as_posix()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    if not normalized or normalized == ".":
        raise ValueError("source_identifier must identify a file")
    return normalized


def content_checksum(content: str) -> str:
    """Hash normalized Unicode text with a stable encoding."""
    normalized_content = normalize("NFC", content)
    return sha256(normalized_content.encode("utf-8")).hexdigest()


def create_document_id(source_identifier: str) -> str:
    """Create a stable UUID for a source's corpus-relative identity."""
    canonical = canonicalize_source_identifier(source_identifier)
    return str(uuid5(INSIGHTFLOW_RAG_NAMESPACE, f"document:{canonical}"))


def create_document_version(normalized_content: str) -> str:
    """Create a content-addressed version for normalized document text."""
    return content_checksum(normalized_content)


def create_element_id(document_id: str, parser_version: str, order: int, text: str) -> str:
    """Create a stable parser-element UUID."""
    if order < 0:
        raise ValueError("element order must be non-negative")
    identity = f"element:{document_id}:{parser_version}:{order}:{content_checksum(text)}"
    return str(uuid5(INSIGHTFLOW_RAG_NAMESPACE, identity))


def create_chunk_id(
    *,
    document_id: str,
    document_version: str,
    chunker_name: str,
    chunker_version: str,
    chunk_index: int,
    content: str,
) -> str:
    """Create a deterministic Qdrant-compatible UUID for one chunk."""
    if chunk_index < 0:
        raise ValueError("chunk_index must be non-negative")
    identity = ":".join(
        (
            "chunk",
            document_id,
            document_version,
            chunker_name,
            chunker_version,
            str(chunk_index),
            content_checksum(content),
        )
    )
    return str(uuid5(INSIGHTFLOW_RAG_NAMESPACE, identity))

