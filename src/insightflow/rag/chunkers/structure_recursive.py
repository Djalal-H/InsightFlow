# src/insightflow/rag/chunkers/structure_recursive.py
"""Structure-aware chunker: splits along document structure, not fixed windows."""

from __future__ import annotations

import re

import tiktoken

from insightflow.rag.config import ChunkingConfig
from insightflow.rag.identity import content_checksum, create_chunk_id
from insightflow.rag.models import Chunk, DocumentElement, NormalizedDocument

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_ENCODING = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    """Conservative token counter shared by all size decisions in this chunker."""
    return len(_ENCODING.encode(text))


def _split_oversized_text(text: str, max_tokens: int) -> list[str]:
    """Break a too-large element into sentence-bounded pieces under max_tokens."""
    sentences = _SENTENCE_SPLIT_RE.split(text)
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = _count_tokens(sentence)
        if current and current_tokens + sentence_tokens > max_tokens:
            pieces.append(" ".join(current))
            current = []
            current_tokens = 0
        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        pieces.append(" ".join(current))
    return pieces or [text]


class _PendingChunk:
    """Accumulator for elements being grouped into one chunk."""

    def __init__(self, heading_path: list[str]) -> None:
        self.heading_path = heading_path
        self.texts: list[str] = []
        self.pages: list[int] = []
        self.token_count = 0

    def add(self, text: str, page_number: int | None, tokens: int) -> None:
        self.texts.append(text)
        if page_number is not None:
            self.pages.append(page_number)
        self.token_count += tokens

    @property
    def content(self) -> str:
        return "\n\n".join(self.texts)


def _group_by_section(elements: list[DocumentElement]) -> list[list[DocumentElement]]:
    """Group contiguous elements that share the same heading_path."""
    sections: list[list[DocumentElement]] = []
    current: list[DocumentElement] = []
    current_path: list[str] | None = None

    for element in elements:
        if element.heading_path != current_path:
            if current:
                sections.append(current)
            current = []
            current_path = element.heading_path
        current.append(element)

    if current:
        sections.append(current)
    return sections


class StructureRecursiveChunker:
    """Split a normalized document along its structure, not fixed windows."""

    strategy_name = "structure_recursive"
    strategy_version = "1.0.0"

    def chunk(self, document: NormalizedDocument, config: ChunkingConfig) -> list[Chunk]:
        pending: list[_PendingChunk] = []

        for section in _group_by_section(document.elements):
            current = _PendingChunk(section[0].heading_path)

            for element in section:
                tokens = _count_tokens(element.text)

                if tokens > config.max_tokens:
                    if current.texts:
                        pending.append(current)
                        current = _PendingChunk(element.heading_path)
                    for piece in _split_oversized_text(element.text, config.max_tokens):
                        piece_tokens = _count_tokens(piece)
                        piece_chunk = _PendingChunk(element.heading_path)
                        piece_chunk.add(piece, element.page_number, piece_tokens)
                        pending.append(piece_chunk)
                    continue

                if current.texts and current.token_count + tokens > config.max_tokens:
                    pending.append(current)
                    current = _PendingChunk(element.heading_path)

                current.add(element.text, element.page_number, tokens)

                if current.token_count >= config.target_tokens:
                    pending.append(current)
                    current = _PendingChunk(element.heading_path)

            if current.texts:
                pending.append(current)

        pending = self._merge_undersized(pending, config)
        return self._finalize(pending, document, config)

    def _merge_undersized(
        self, pending: list[_PendingChunk], config: ChunkingConfig
    ) -> list[_PendingChunk]:
        """Fold chunks below min_tokens into a neighbor when it keeps size in bounds."""
        merged: list[_PendingChunk] = []
        for piece in pending:
            if (
                merged
                and piece.token_count < config.min_tokens
                and merged[-1].heading_path == piece.heading_path
                and merged[-1].token_count + piece.token_count <= config.max_tokens
            ):
                merged[-1].texts.extend(piece.texts)
                merged[-1].pages.extend(piece.pages)
                merged[-1].token_count += piece.token_count
            else:
                merged.append(piece)
        return merged

    def _finalize(
        self,
        pending: list[_PendingChunk],
        document: NormalizedDocument,
        config: ChunkingConfig,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        chunk_ids: list[str] = []

        for index, piece in enumerate(pending):
            content = piece.content
            chunk_id = create_chunk_id(
                document_id=document.document_id,
                document_version=document.document_version,
                chunker_name=self.strategy_name,
                chunker_version=self.strategy_version,
                chunk_index=index,
                content=content,
            )
            chunk_ids.append(chunk_id)

            page_start = min(piece.pages) if piece.pages else None
            page_end = max(piece.pages) if piece.pages else None

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    document_version=document.document_version,
                    chunk_index=index,
                    content=content,
                    embedding_text=content,
                    heading_path=piece.heading_path,
                    page_start=page_start,
                    page_end=page_end,
                    token_count=max(piece.token_count, 1),
                    content_type="text",
                    content_checksum=content_checksum(content),
                    chunker_name=self.strategy_name,
                    chunker_version=self.strategy_version,
                )
            )

        for index, chunk in enumerate(chunks):
            previous_id = chunk_ids[index - 1] if index > 0 else None
            next_id = chunk_ids[index + 1] if index < len(chunks) - 1 else None
            chunks[index] = chunk.model_copy(
                update={"previous_chunk_id": previous_id, "next_chunk_id": next_id}
            )

        return chunks