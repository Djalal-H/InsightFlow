"""Tests for query expansion and rank fusion."""

from __future__ import annotations

import json

import pytest

from insightflow.rag.models import RetrievalQuery
from insightflow.rag.query_expansion import (
    DECOMPOSE_SYSTEM_PROMPT,
    MULTI_QUERY_SYSTEM_PROMPT,
    QueryExpandingRetriever,
    reciprocal_rank_fusion,
)
from tests.fakes import ScriptedChatProvider, StaticRetriever, make_candidate


def queries(*texts: str) -> str:
    return json.dumps({"queries": list(texts)})


def test_fusion_prefers_chunks_that_several_rankings_return() -> None:
    a, b, c = (make_candidate(name, rank=1) for name in ("a", "b", "c"))

    fused = reciprocal_rank_fusion([[a, b], [c, b]], limit=3)

    assert [item.chunk.chunk_id for item in fused] == ["b", "a", "c"]
    assert [item.rank for item in fused] == [1, 2, 3]
    assert fused[0].score == pytest.approx(2 / 62)
    assert fused[1].score == pytest.approx(1 / 61)


def test_fusion_applies_the_limit() -> None:
    ranking = [make_candidate(name) for name in ("a", "b", "c")]

    assert len(reciprocal_rank_fusion([ranking], limit=2)) == 2


async def test_multi_query_searches_each_wording_and_fuses() -> None:
    chat = ScriptedChatProvider(expand=queries("tuition cost", "fees per semester"))
    inner = StaticRetriever(
        by_query={
            "How much is tuition?": [make_candidate("a"), make_candidate("b", rank=2)],
            "tuition cost": [make_candidate("b"), make_candidate("c", rank=2)],
            "fees per semester": [make_candidate("b")],
        }
    )
    retriever = QueryExpandingRetriever(inner, chat, mode="multi_query", max_queries=3)

    results = await retriever.retrieve(
        RetrievalQuery(text="How much is tuition?", top_k=2, source_identifiers=["x.pdf"])
    )

    assert [query.text for query in inner.queries] == [
        "How much is tuition?",
        "tuition cost",
        "fees per semester",
    ]
    assert all(query.source_identifiers == ["x.pdf"] for query in inner.queries)
    assert [item.chunk.chunk_id for item in results] == ["b", "a"]
    [messages] = chat.calls_for("expand")
    assert messages[0]["content"] == MULTI_QUERY_SYSTEM_PROMPT
    assert "at most 2 queries" in messages[1]["content"]


async def test_expansion_drops_duplicates_blanks_and_extra_queries() -> None:
    chat = ScriptedChatProvider(expand=queries("what is x?", " ", "x meaning", "x meaning", "y"))
    retriever = QueryExpandingRetriever(
        StaticRetriever(), chat, mode="multi_query", max_queries=3
    )

    assert await retriever.expand("What is X?") == ["What is X?", "x meaning", "y"]


@pytest.mark.parametrize("reply", ["not json", queries()])
async def test_unusable_expansion_searches_the_original_query_only(reply: str) -> None:
    inner = StaticRetriever([make_candidate("a")])
    retriever = QueryExpandingRetriever(
        inner, ScriptedChatProvider(expand=reply), mode="decompose"
    )

    results = await retriever.retrieve(RetrievalQuery(text="One fact?"))

    assert [query.text for query in inner.queries] == ["One fact?"]
    assert results == inner.candidates


async def test_decompose_uses_its_own_prompt() -> None:
    chat = ScriptedChatProvider(expand=queries("dean of nursing", "dean of history"))
    inner = StaticRetriever(
        by_query={
            "dean of nursing": [make_candidate("n")],
            "dean of history": [make_candidate("h")],
        },
        candidates=[],
    )
    retriever = QueryExpandingRetriever(inner, chat, mode="decompose")

    results = await retriever.retrieve(
        RetrievalQuery(text="Who are the deans of nursing and history?")
    )

    assert chat.calls_for("expand")[0][0]["content"] == DECOMPOSE_SYSTEM_PROMPT
    assert {item.chunk.chunk_id for item in results} == {"n", "h"}


def test_expansion_needs_room_for_one_more_query() -> None:
    with pytest.raises(ValueError, match="max_queries"):
        QueryExpandingRetriever(
            StaticRetriever(), ScriptedChatProvider(), mode="multi_query", max_queries=1
        )
