"""Bounded Max-Min semantic chunking over normalized document elements."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import cast

from pydantic import JsonValue

from insightflow.core.exceptions import ProviderError
from insightflow.providers.embeddings import EmbeddingProvider
from insightflow.rag.config import ChunkingConfig
from insightflow.rag.identity import content_checksum, create_chunk_id
from insightflow.rag.models import Chunk, DocumentElement, ElementType, NormalizedDocument

_ATOMIC_ELEMENT_TYPES = frozenset({"title", "heading", "table", "caption", "code"})
_CLOSING_PUNCTUATION = frozenset({'"', "'", "’", "”", ")", "]", "}"})
_ENGLISH_ABBREVIATIONS = frozenset(
    {
        "e.g.",
        "etc.",
        "i.e.",
        "jr.",
        "mr.",
        "mrs.",
        "ms.",
        "prof.",
        "sr.",
        "st.",
        "vs.",
        "dr.",
        "fig.",
        "no.",
        "u.s.",
    }
)


@dataclass(frozen=True)
class _TextSpan:
    start: int
    end: int


@dataclass(frozen=True)
class _Unit:
    """One contiguous semantic input with source provenance."""

    index: int
    text: str
    separator_before: str
    element_id: str
    element_order: int
    element_type: ElementType
    heading_path: tuple[str, ...]
    page_number: int | None
    forced_token_split: bool = False
    embedding: tuple[float, ...] = ()


def _trimmed_span(text: str, start: int, end: int) -> _TextSpan | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return _TextSpan(start, end) if start < end else None


def _is_abbreviation(text: str, period_index: int) -> bool:
    if (
        period_index > 0
        and period_index + 1 < len(text)
        and text[period_index - 1].isdigit()
        and text[period_index + 1].isdigit()
    ):
        return True
    prefix = text[: period_index + 1]
    match = re.search(r"([A-Za-z](?:[A-Za-z.]*)\.)$", prefix)
    if match is None:
        return False
    value = match.group(1).lower()
    return value in _ENGLISH_ABBREVIATIONS or bool(re.fullmatch(r"[a-z]\.", value))


def _sentence_spans(text: str) -> list[_TextSpan]:
    """Return deterministic English-first sentence spans without model downloads."""
    spans: list[_TextSpan] = []
    block_start = 0
    block_boundaries = [match.span() for match in re.finditer(r"\n\s*\n", text)]
    blocks = [*block_boundaries, (len(text), len(text))]
    for boundary_start, boundary_end in blocks:
        block_end = boundary_start
        sentence_start = block_start
        index = block_start
        while index < block_end:
            character = text[index]
            if character in ".!?" and not (
                character == "." and _is_abbreviation(text, index)
            ):
                candidate_end = index + 1
                while (
                    candidate_end < block_end
                    and text[candidate_end] in _CLOSING_PUNCTUATION
                ):
                    candidate_end += 1
                if candidate_end == block_end or text[candidate_end].isspace():
                    span = _trimmed_span(text, sentence_start, candidate_end)
                    if span is not None:
                        spans.append(span)
                    sentence_start = candidate_end
                    index = candidate_end
                    continue
            index += 1
        span = _trimmed_span(text, sentence_start, block_end)
        if span is not None:
            spans.append(span)
        block_start = boundary_end
    return spans


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot / (left_norm * right_norm)


class MaxMinSemanticChunker:
    """Create bounded, traceable chunks using sentence-level Max-Min clustering."""

    strategy_name = "semantic_max_min"
    strategy_version = "1"

    def __init__(self, embedding_provider: EmbeddingProvider, *, embedding_batch_size: int = 32):
        if embedding_batch_size < 1:
            raise ValueError("embedding_batch_size must be positive")
        self._embedding_provider = embedding_provider
        self._embedding_batch_size = embedding_batch_size
        self._token_count_cache: dict[str, int] = {}

    async def chunk(
        self,
        document: NormalizedDocument,
        config: ChunkingConfig,
    ) -> list[Chunk]:
        """Split a normalized document using hosted embeddings and bounded Max-Min rules."""
        if config.strategy != self.strategy_name:
            raise ValueError(f"{self.strategy_name} chunker requires its matching strategy")

        self._token_count_cache.clear()
        units = self._create_units(document.elements, config.max_tokens)
        embedded_units = await self._embed_units(units)
        semantic_clusters = self._cluster_units(embedded_units, config)
        bounded = [
            split
            for cluster in semantic_clusters
            for split in self._split_oversized_cluster(cluster, config)
        ]
        bounded = self._merge_undersized_clusters(bounded, config)
        return self._create_chunks(document, bounded, config)

    def _create_units(
        self,
        elements: Sequence[DocumentElement],
        max_tokens: int,
    ) -> list[_Unit]:
        units: list[_Unit] = []
        previous_element: DocumentElement | None = None
        previous_end = 0
        for element in elements:
            spans = (
                [_trimmed_span(element.text, 0, len(element.text))]
                if element.element_type in _ATOMIC_ELEMENT_TYPES
                else _sentence_spans(element.text)
            )
            for possible_span in spans:
                if possible_span is None:
                    continue
                span = possible_span
                if previous_element is element:
                    separator = element.text[previous_end : span.start]
                elif previous_element is None:
                    separator = ""
                else:
                    separator = "\n\n"
                unit = _Unit(
                    index=len(units),
                    text=element.text[span.start : span.end],
                    separator_before=separator,
                    element_id=element.element_id,
                    element_order=element.order,
                    element_type=element.element_type,
                    heading_path=tuple(element.heading_path),
                    page_number=element.page_number,
                )
                split_units = self._split_oversized_unit(unit, max_tokens)
                units.extend(
                    replace(value, index=len(units) + offset)
                    for offset, value in enumerate(split_units)
                )
                previous_element = element
                previous_end = span.end
        return units

    def _split_oversized_unit(self, unit: _Unit, max_tokens: int) -> list[_Unit]:
        if self._count_tokens(unit.text) <= max_tokens:
            return [unit]

        spans: list[_TextSpan] = []
        start = 0
        text = unit.text
        while start < len(text):
            candidate_ends = [
                match.end()
                for match in re.finditer(r"\S+\s*", text[start:])
            ]
            best_end: int | None = None
            for relative_end in candidate_ends:
                end = start + relative_end
                candidate = text[start:end].strip()
                if candidate and self._count_tokens(candidate) <= max_tokens:
                    best_end = end
                elif best_end is not None:
                    break
            if best_end is None:
                best_end = self._largest_character_prefix(text, start, max_tokens)
            span = _trimmed_span(text, start, best_end)
            if span is None:
                raise ProviderError("Token counting could not produce a bounded semantic unit")
            spans.append(span)
            start = best_end

        split: list[_Unit] = []
        previous_end = 0
        for index, span in enumerate(spans):
            separator = unit.separator_before if index == 0 else text[previous_end : span.start]
            split.append(
                replace(
                    unit,
                    text=text[span.start : span.end],
                    separator_before=separator,
                    forced_token_split=True,
                )
            )
            previous_end = span.end
        return split

    def _largest_character_prefix(self, text: str, start: int, max_tokens: int) -> int:
        low = start + 1
        high = len(text)
        best: int | None = None
        while low <= high:
            middle = (low + high) // 2
            candidate = text[start:middle].strip()
            if candidate and self._count_tokens(candidate) <= max_tokens:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        if best is None:
            raise ProviderError("A single character exceeds the configured chunk token maximum")
        return best

    async def _embed_units(self, units: Sequence[_Unit]) -> list[_Unit]:
        embedded: list[_Unit] = []
        expected_dimension: int | None = None
        for start in range(0, len(units), self._embedding_batch_size):
            batch = units[start : start + self._embedding_batch_size]
            vectors = await self._embedding_provider.embed_documents(
                [unit.text for unit in batch]
            )
            if len(vectors) != len(batch):
                raise ProviderError("The embedding provider returned an unexpected vector count")
            for unit, vector in zip(batch, vectors, strict=True):
                validated = self._validate_vector(vector, expected_dimension)
                expected_dimension = len(validated)
                embedded.append(replace(unit, embedding=validated))
        return embedded

    @staticmethod
    def _validate_vector(
        vector: Sequence[float],
        expected_dimension: int | None,
    ) -> tuple[float, ...]:
        try:
            values = tuple(float(value) for value in vector)
        except (TypeError, ValueError) as exc:
            raise ProviderError("The embedding provider returned an invalid vector") from exc
        if not values or any(not math.isfinite(value) for value in values):
            raise ProviderError("The embedding provider returned an invalid vector")
        if expected_dimension is not None and len(values) != expected_dimension:
            raise ProviderError("The embedding provider returned inconsistent vector dimensions")
        if not any(value != 0.0 for value in values):
            raise ProviderError("The embedding provider returned a zero-length vector")
        return values

    @staticmethod
    def _cluster_units(units: Sequence[_Unit], config: ChunkingConfig) -> list[list[_Unit]]:
        if not units:
            return []
        clusters: list[list[_Unit]] = []
        current = [units[0]]
        pairwise_min: float | None = None
        for candidate in units[1:]:
            similarities = [_cosine(candidate.embedding, unit.embedding) for unit in current]
            if len(current) == 1:
                score = config.initialization_constant * similarities[0]
                threshold = config.hard_threshold
            else:
                if pairwise_min is None:
                    raise RuntimeError("Max-Min cluster similarity was not initialized")
                score = max(similarities)
                threshold = max(
                    config.hard_threshold,
                    pairwise_min
                    * config.similarity_coefficient
                    * _sigmoid(float(len(current) - 1)),
                )
            if score > threshold:
                current.append(candidate)
                candidate_min = min(similarities)
                pairwise_min = (
                    candidate_min if pairwise_min is None else min(pairwise_min, candidate_min)
                )
            else:
                clusters.append(current)
                current = [candidate]
                pairwise_min = None
        clusters.append(current)
        return clusters

    def _split_oversized_cluster(
        self,
        cluster: list[_Unit],
        config: ChunkingConfig,
    ) -> list[list[_Unit]]:
        if self._cluster_token_count(cluster) <= config.max_tokens:
            return [cluster]
        chunks: list[list[_Unit]] = []
        start = 0
        while start < len(cluster):
            if self._cluster_token_count(cluster[start:]) <= config.max_tokens:
                chunks.append(cluster[start:])
                break
            fitting: list[tuple[int, int]] = []
            for end in range(start + 1, len(cluster) + 1):
                token_count = self._cluster_token_count(cluster[start:end])
                if token_count <= config.max_tokens:
                    fitting.append((end, token_count))
            if not fitting:
                raise ProviderError("A semantic unit exceeds the configured chunk token maximum")
            preferred = [item for item in fitting if item[1] >= config.target_tokens]
            if preferred:
                end, _ = min(
                    preferred,
                    key=lambda item: (
                        _cosine(cluster[item[0] - 1].embedding, cluster[item[0]].embedding),
                        abs(config.target_tokens - item[1]),
                        item[0],
                    ),
                )
            else:
                end, _ = max(fitting, key=lambda item: (item[1], item[0]))
            chunks.append(cluster[start:end])
            start = end
        return chunks

    def _merge_undersized_clusters(
        self,
        clusters: list[list[_Unit]],
        config: ChunkingConfig,
    ) -> list[list[_Unit]]:
        while len(clusters) > 1:
            candidates: list[tuple[float, int, int, int]] = []
            for index, cluster in enumerate(clusters):
                if self._cluster_token_count(cluster) >= config.min_tokens:
                    continue
                if index > 0:
                    combined = clusters[index - 1] + cluster
                    if self._cluster_token_count(combined) <= config.max_tokens:
                        candidates.append(
                            (
                                self._cluster_affinity(clusters[index - 1], cluster),
                                1,
                                index,
                                -1,
                            )
                        )
                if index + 1 < len(clusters):
                    combined = cluster + clusters[index + 1]
                    if self._cluster_token_count(combined) <= config.max_tokens:
                        candidates.append(
                            (
                                self._cluster_affinity(cluster, clusters[index + 1]),
                                0,
                                index,
                                1,
                            )
                        )
            if not candidates:
                break
            _, _, index, direction = max(candidates)
            if direction < 0:
                clusters[index - 1 : index + 1] = [clusters[index - 1] + clusters[index]]
            else:
                clusters[index : index + 2] = [clusters[index] + clusters[index + 1]]
        return clusters

    @staticmethod
    def _cluster_affinity(left: Sequence[_Unit], right: Sequence[_Unit]) -> float:
        return max(
            _cosine(left_unit.embedding, right_unit.embedding)
            for left_unit in left
            for right_unit in right
        )

    def _create_chunks(
        self,
        document: NormalizedDocument,
        clusters: Sequence[Sequence[_Unit]],
        config: ChunkingConfig,
    ) -> list[Chunk]:
        contents = [self._render(cluster) for cluster in clusters]
        chunk_ids = [
            create_chunk_id(
                document_id=document.document_id,
                document_version=document.document_version,
                chunker_name=self.strategy_name,
                chunker_version=self.strategy_version,
                chunk_index=index,
                content=content,
            )
            for index, content in enumerate(contents)
        ]
        chunks: list[Chunk] = []
        for index, (cluster, content, chunk_id) in enumerate(
            zip(clusters, contents, chunk_ids, strict=True)
        ):
            token_count = self._count_tokens(content)
            page_numbers = [unit.page_number for unit in cluster if unit.page_number is not None]
            element_types = list(dict.fromkeys(unit.element_type for unit in cluster))
            element_ids = list(dict.fromkeys(unit.element_id for unit in cluster))
            element_orders = list(dict.fromkeys(unit.element_order for unit in cluster))
            metadata: dict[str, JsonValue] = {
                "element_ids": cast(JsonValue, element_ids),
                "element_orders": cast(JsonValue, element_orders),
                "unit_start": cluster[0].index,
                "unit_end": cluster[-1].index,
                "forced_token_split": any(unit.forced_token_split for unit in cluster),
                "below_min_tokens": token_count < config.min_tokens,
                "max_min": cast(
                    JsonValue,
                    {
                        "hard_threshold": config.hard_threshold,
                        "similarity_coefficient": config.similarity_coefficient,
                        "initialization_constant": config.initialization_constant,
                    },
                ),
            }
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    document_version=document.document_version,
                    chunk_index=index,
                    content=content,
                    embedding_text=content,
                    heading_path=self._common_heading_path(cluster),
                    page_start=min(page_numbers) if page_numbers else None,
                    page_end=max(page_numbers) if page_numbers else None,
                    token_count=token_count,
                    content_type=(element_types[0] if len(element_types) == 1 else "mixed"),
                    previous_chunk_id=chunk_ids[index - 1] if index > 0 else None,
                    next_chunk_id=chunk_ids[index + 1] if index + 1 < len(chunk_ids) else None,
                    content_checksum=content_checksum(content),
                    chunker_name=self.strategy_name,
                    chunker_version=self.strategy_version,
                    metadata=metadata,
                )
            )
        return chunks

    @staticmethod
    def _common_heading_path(cluster: Sequence[_Unit]) -> list[str]:
        if not cluster:
            return []
        common = list(cluster[0].heading_path)
        for unit in cluster[1:]:
            prefix_length = 0
            for left, right in zip(common, unit.heading_path, strict=False):
                if left != right:
                    break
                prefix_length += 1
            common = common[:prefix_length]
            if not common:
                break
        return common

    def _cluster_token_count(self, cluster: Sequence[_Unit]) -> int:
        return self._count_tokens(self._render(cluster))

    @staticmethod
    def _render(cluster: Sequence[_Unit]) -> str:
        if not cluster:
            return ""
        parts = [cluster[0].text]
        for unit in cluster[1:]:
            parts.extend((unit.separator_before, unit.text))
        return "".join(parts)

    def _count_tokens(self, text: str) -> int:
        cached = self._token_count_cache.get(text)
        if cached is not None:
            return cached
        count = self._embedding_provider.count_tokens(text)
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ProviderError("Token counting returned an invalid result")
        self._token_count_cache[text] = count
        return count
