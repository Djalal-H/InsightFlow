"""Candidate processors that refine retrieved passages before context assembly.

- `Reranker` orders the candidates with a hosted reranking model and keeps the best.
- `NeighborExpander` gives the model the chunks around a hit (parent context), read from the
  links that the chunkers store.
- `ContextCompressor` keeps only the sentences that a chat model selects for the question.

`RefiningRetriever` runs processors over the result of another retriever, so the workflow only
knows the `Retriever` protocol.
"""

import re
from collections.abc import Callable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from insightflow.providers.llm import ChatProvider
from insightflow.providers.rerank import RerankProvider
from insightflow.providers.structured import complete_json
from insightflow.rag.models import Chunk, RetrievalQuery, RetrievedChunk
from insightflow.rag.protocols import CandidateProcessor, ChunkReader, Retriever

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_UNCOMPRESSED_CONTENT_TYPES = frozenset({"table", "code"})
OMISSION_MARK = "[...]"
MIN_SENTENCES_TO_COMPRESS = 3
COMPRESSOR_SYSTEM_PROMPT = (
    "You select the sentences of passages that help to answer a question. Return only a JSON "
    'object: {"keep": {"<passage number>": [<sentence numbers>]}}. '
    "Keep each sentence that holds a fact for the answer, and each sentence that is necessary "
    "to understand such a fact. Use an empty list for a passage in which no sentence helps. "
    "Do not answer the question."
)


class RefiningRetriever:
    """Retrieve with `inner`, then apply each processor in order."""

    def __init__(self, inner: Retriever, processors: Sequence[CandidateProcessor]) -> None:
        self._inner = inner
        self._processors = tuple(processors)

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        candidates = await self._inner.retrieve(query)
        for processor in self._processors:
            if not candidates:
                break
            candidates = await processor.process(query, candidates)
        return candidates


class Reranker:
    """Order candidates by a hosted reranker's relevance score and keep the best `top_n`."""

    def __init__(self, provider: RerankProvider, *, top_n: int) -> None:
        if top_n < 1:
            raise ValueError("top_n must be at least 1")
        self._provider = provider
        self._top_n = top_n

    async def process(
        self, query: RetrievalQuery, candidates: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        results = await self._provider.rerank(
            query.text,
            [candidate.chunk.content for candidate in candidates],
            top_n=self._top_n,
        )
        return [
            candidates[result.index].model_copy(update={"score": result.score, "rank": rank})
            for rank, result in enumerate(results[: self._top_n], start=1)
        ]


class NeighborExpander:
    """Add the neighboring chunks of the best hits to their content.

    Only the first `limit` candidates are expanded, because only they can enter the context.
    A neighbor that is itself a candidate, or that an earlier hit already took, is not added
    again. The passage keeps the identity of the hit; `metadata["expanded_chunk_ids"]` lists
    the neighbors.
    """

    def __init__(self, reader: ChunkReader, *, neighbors: int, limit: int) -> None:
        if neighbors < 1 or limit < 1:
            raise ValueError("neighbors and limit must be at least 1")
        self._reader = reader
        self._neighbors = neighbors
        self._limit = limit

    async def process(
        self, query: RetrievalQuery, candidates: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        head = candidates[: self._limit]
        windows: list[list[Chunk]] = [[candidate.chunk] for candidate in head]
        open_sides = [[True, True] for _ in head]
        claimed = {candidate.chunk.chunk_id for candidate in candidates}

        for _ in range(self._neighbors):
            wanted: list[tuple[int, int, str]] = []
            for position, window in enumerate(windows):
                for side, neighbor_id in (
                    (0, window[0].previous_chunk_id),
                    (1, window[-1].next_chunk_id),
                ):
                    if not open_sides[position][side]:
                        continue
                    if neighbor_id is None or neighbor_id in claimed:
                        open_sides[position][side] = False
                        continue
                    claimed.add(neighbor_id)
                    wanted.append((position, side, neighbor_id))
            if not wanted:
                break
            found = {
                chunk.chunk_id: chunk
                for chunk in await self._reader.get_chunks([item[2] for item in wanted])
            }
            for position, side, neighbor_id in wanted:
                neighbor = found.get(neighbor_id)
                if neighbor is None:
                    open_sides[position][side] = False
                elif side == 0:
                    windows[position].insert(0, neighbor)
                else:
                    windows[position].append(neighbor)

        expanded = [
            candidate.model_copy(update={"chunk": _merge_window(candidate.chunk, window)})
            for candidate, window in zip(head, windows, strict=True)
        ]
        return [*expanded, *candidates[self._limit :]]


def _merge_window(hit: Chunk, window: list[Chunk]) -> Chunk:
    if len(window) == 1:
        return hit
    pages_start = [chunk.page_start for chunk in window if chunk.page_start is not None]
    pages_end = [chunk.page_end for chunk in window if chunk.page_end is not None]
    return hit.model_copy(
        update={
            "content": "\n\n".join(chunk.content for chunk in window),
            "token_count": sum(chunk.token_count for chunk in window),
            "page_start": min(pages_start) if pages_start else None,
            "page_end": max(pages_end) if pages_end else None,
            "metadata": {
                **hit.metadata,
                "expanded_chunk_ids": [
                    chunk.chunk_id for chunk in window if chunk.chunk_id != hit.chunk_id
                ],
            },
        }
    )


class SentenceSelection(BaseModel):
    """Chat-model output for context compression."""

    model_config = ConfigDict(extra="ignore")

    keep: dict[str, list[int]] = Field(default_factory=dict)


class ContextCompressor:
    """Cut each of the first `limit` passages to the sentences that the model selects.

    The kept sentences are the passage's own words, in order; `[...]` marks an omission.
    A passage stays complete when it is short, is a table or code, has no valid selection, or
    has an empty one. Thus a wrong or unreadable model reply never removes a passage.
    """

    def __init__(
        self,
        chat_provider: ChatProvider,
        count_tokens: Callable[[str], int],
        *,
        limit: int,
    ) -> None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        self._chat_provider = chat_provider
        self._count_tokens = count_tokens
        self._limit = limit

    async def process(
        self, query: RetrievalQuery, candidates: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        sentences_by_position: dict[int, list[str]] = {}
        for position, candidate in enumerate(candidates[: self._limit]):
            if candidate.chunk.content_type in _UNCOMPRESSED_CONTENT_TYPES:
                continue
            sentences = split_sentences(candidate.chunk.content)
            if len(sentences) >= MIN_SENTENCES_TO_COMPRESS:
                sentences_by_position[position] = sentences
        if not sentences_by_position:
            return candidates

        blocks = [
            f"Passage {position + 1}:\n"
            + "\n".join(f"({number}) {text}" for number, text in enumerate(sentences, start=1))
            for position, sentences in sentences_by_position.items()
        ]
        messages = [
            {"role": "system", "content": COMPRESSOR_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {query.text}\n\n" + "\n\n".join(blocks)},
        ]
        selection = await complete_json(self._chat_provider, messages, SentenceSelection)
        if selection is None:
            return candidates

        compressed = list(candidates)
        for position, sentences in sentences_by_position.items():
            kept = sorted(
                {
                    number
                    for number in selection.keep.get(str(position + 1), [])
                    if 1 <= number <= len(sentences)
                }
            )
            if not kept or len(kept) == len(sentences):
                continue
            content = _join_kept(sentences, kept)
            chunk = candidates[position].chunk
            compressed[position] = candidates[position].model_copy(
                update={
                    "chunk": chunk.model_copy(
                        update={
                            "content": content,
                            "token_count": max(1, self._count_tokens(content)),
                            "metadata": {**chunk.metadata, "compressed": True},
                        }
                    )
                }
            )
        return compressed


def split_sentences(text: str) -> list[str]:
    """Split text at sentence ends and line breaks; empty pieces are dropped."""
    return [
        sentence.strip()
        for line in text.splitlines()
        for sentence in _SENTENCE_SPLIT_RE.split(line)
        if sentence.strip()
    ]


def _join_kept(sentences: list[str], kept: list[int]) -> str:
    parts: list[str] = []
    previous = 0
    for number in kept:
        if number != previous + 1:
            parts.append(OMISSION_MARK)
        parts.append(sentences[number - 1])
        previous = number
    if previous != len(sentences):
        parts.append(OMISSION_MARK)
    return " ".join(parts)
