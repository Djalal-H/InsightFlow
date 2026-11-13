"""Tests for the candidate processors: reranking, neighbor expansion, and compression."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest

from insightflow.providers.rerank import RerankResult
from insightflow.rag.models import Chunk, RetrievalQuery, RetrievedChunk
from insightflow.rag.refinement import (
    ContextCompressor,
    NeighborExpander,
    RefiningRetriever,
    Reranker,
    split_sentences,
)
from tests.fakes import (
    FakeRerankProvider,
    ScriptedChatProvider,
    StaticRetriever,
    make_candidate,
    word_count,
)

QUERY = RetrievalQuery(text="How deep is the river?")


def ids(candidates: Sequence[RetrievedChunk]) -> list[str]:
    return [candidate.chunk.chunk_id for candidate in candidates]


# --- reranking ---------------------------------------------------------------------------


async def test_reranker_orders_by_provider_score_and_keeps_top_n() -> None:
    candidates = [
        make_candidate("a", "The desert is dry.", rank=1),
        make_candidate("b", "The river is deep. The river is wide.", rank=2),
        make_candidate("c", "A river.", rank=3),
    ]
    provider = FakeRerankProvider("river")

    results = await Reranker(provider, top_n=2).process(QUERY, candidates)

    assert ids(results) == ["b", "c"]
    assert [item.rank for item in results] == [1, 2]
    assert [item.score for item in results] == [2.0, 1.0]
    assert provider.calls == [
        ("How deep is the river?", [item.chunk.content for item in candidates], 2)
    ]


async def test_reranker_truncates_a_provider_that_returns_too_many() -> None:
    class Generous:
        async def rerank(
            self, query: str, documents: Sequence[str], *, top_n: int
        ) -> list[RerankResult]:
            return [RerankResult(index=i, score=1.0) for i in range(len(documents))]

    candidates = [make_candidate(name) for name in ("a", "b", "c")]

    assert ids(await Reranker(Generous(), top_n=1).process(QUERY, candidates)) == ["a"]
    with pytest.raises(ValueError, match="top_n"):
        Reranker(Generous(), top_n=0)


async def test_refining_retriever_runs_processors_in_order_and_skips_empty_results() -> None:
    provider = FakeRerankProvider("river")
    candidates = [make_candidate("a", "desert"), make_candidate("b", "river", rank=2)]

    refined = RefiningRetriever(StaticRetriever(candidates), [Reranker(provider, top_n=1)])
    empty = RefiningRetriever(StaticRetriever([]), [Reranker(provider, top_n=1)])

    assert ids(await refined.retrieve(QUERY)) == ["b"]
    assert await empty.retrieve(QUERY) == []
    assert len(provider.calls) == 1


# --- neighbor expansion ------------------------------------------------------------------


class ChunkReader:
    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self.requests: list[list[str]] = []

    async def get_chunks(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        self.requests.append(list(chunk_ids))
        return [self.chunks[chunk_id] for chunk_id in chunk_ids if chunk_id in self.chunks]


def document_chunks(count: int, **fields: Any) -> list[RetrievedChunk]:
    """A document of linked chunks c0..c(n-1); chunk i is on page i + 1."""
    names = [f"c{index}" for index in range(count)]
    return [
        make_candidate(
            name,
            f"Text {index}.",
            rank=index + 1,
            previous_chunk_id=names[index - 1] if index > 0 else None,
            next_chunk_id=names[index + 1] if index < count - 1 else None,
            page_start=index + 1,
            page_end=index + 1,
            **fields,
        )
        for index, name in enumerate(names)
    ]


async def test_expander_adds_both_neighbors_and_keeps_the_hit_identity() -> None:
    document = document_chunks(5)
    reader = ChunkReader([item.chunk for item in document])
    hit = document[2].model_copy(update={"rank": 1})

    [expanded] = await NeighborExpander(reader, neighbors=1, limit=3).process(QUERY, [hit])

    assert expanded.chunk.chunk_id == "c2"
    assert expanded.chunk.content == "Text 1.\n\nText 2.\n\nText 3."
    assert (expanded.chunk.page_start, expanded.chunk.page_end) == (2, 4)
    assert expanded.chunk.token_count == 6
    assert expanded.chunk.metadata["expanded_chunk_ids"] == ["c1", "c3"]
    assert expanded.score == hit.score
    assert reader.requests == [["c1", "c3"]]


async def test_expander_walks_two_neighbors_and_stops_at_document_edges() -> None:
    document = document_chunks(4)
    reader = ChunkReader([item.chunk for item in document])
    hit = document[1].model_copy(update={"rank": 1})

    [expanded] = await NeighborExpander(reader, neighbors=2, limit=1).process(QUERY, [hit])

    assert expanded.chunk.content == "Text 0.\n\nText 1.\n\nText 2.\n\nText 3."
    assert reader.requests == [["c0", "c2"], ["c3"]]


async def test_expander_does_not_repeat_candidates_or_claimed_neighbors() -> None:
    document = document_chunks(5)
    reader = ChunkReader([item.chunk for item in document])
    # c1 and c3 are hits; c2 lies between them and c4 is a lower-ranked candidate.
    candidates = [
        document[1].model_copy(update={"rank": 1}),
        document[3].model_copy(update={"rank": 2}),
        document[4].model_copy(update={"rank": 3}),
    ]

    results = await NeighborExpander(reader, neighbors=1, limit=2).process(QUERY, candidates)

    assert ids(results) == ["c1", "c3", "c4"]
    assert results[0].chunk.content == "Text 0.\n\nText 1.\n\nText 2."
    assert results[1].chunk.content == "Text 3."  # c2 was taken, c4 is a candidate
    assert results[2] is candidates[2]  # below the limit: not expanded


async def test_expander_leaves_hits_without_stored_neighbors_unchanged() -> None:
    hit = make_candidate("c1", previous_chunk_id="missing")

    [result] = await NeighborExpander(ChunkReader([]), neighbors=1, limit=1).process(
        QUERY, [hit]
    )

    assert result == hit
    with pytest.raises(ValueError, match="at least 1"):
        NeighborExpander(ChunkReader([]), neighbors=0, limit=1)


# --- compression -------------------------------------------------------------------------

LONG = "The river is ten metres deep. It was mapped in 1990. Many fish live there. It is cold."


def keep(**selection: list[int]) -> str:
    return json.dumps({"keep": {name.removeprefix("p"): numbers for name, numbers in
                                selection.items()}})


def test_split_sentences_uses_sentence_ends_and_lines() -> None:
    assert split_sentences("One. Two!\nThree\n\nFour?") == ["One.", "Two!", "Three", "Four?"]


async def test_compressor_keeps_selected_sentences_and_marks_omissions() -> None:
    chat = ScriptedChatProvider(compress=keep(p1=[1, 3]))
    candidates = [make_candidate("a", LONG), make_candidate("b", "Short. Text.", rank=2)]

    results = await ContextCompressor(chat, word_count, limit=6).process(QUERY, candidates)

    compressed = results[0].chunk
    assert compressed.content == (
        "The river is ten metres deep. [...] Many fish live there. [...]"
    )
    assert compressed.token_count == word_count(compressed.content)
    assert compressed.metadata["compressed"] is True
    assert compressed.chunk_id == "a"
    assert results[1] is candidates[1]
    prompt = chat.calls_for("compress")[0][1]["content"]
    assert prompt.startswith("Question: How deep is the river?\n\nPassage 1:\n(1) The river")
    assert "Passage 2" not in prompt  # too short to compress


@pytest.mark.parametrize(
    "reply",
    ["not json", keep(p1=[]), keep(p1=[1, 2, 3, 4]), keep(p1=[0, 9]), keep(p2=[1])],
)
async def test_compressor_never_drops_or_empties_a_passage(reply: str) -> None:
    candidates = [make_candidate("a", LONG)]

    results = await ContextCompressor(
        ScriptedChatProvider(compress=reply), word_count, limit=6
    ).process(QUERY, candidates)

    assert results == candidates


async def test_compressor_skips_tables_code_and_passages_beyond_the_limit() -> None:
    chat = ScriptedChatProvider(compress=keep(p1=[1], p2=[1], p3=[1]))
    candidates = [
        make_candidate("t", LONG, content_type="table"),
        make_candidate("a", LONG, rank=2),
        make_candidate("b", LONG, rank=3),
    ]

    results = await ContextCompressor(chat, word_count, limit=2).process(QUERY, candidates)

    assert results[0] is candidates[0]
    assert results[1].chunk.content == "The river is ten metres deep. [...]"
    assert results[2] is candidates[2]


async def test_compressor_makes_no_call_without_long_passages() -> None:
    chat = ScriptedChatProvider()
    candidates = [make_candidate("a", "Short text.")]

    assert await ContextCompressor(chat, word_count, limit=6).process(QUERY, candidates) == (
        candidates
    )
    assert chat.calls == []
