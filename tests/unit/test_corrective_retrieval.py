"""Tests for corrective retrieval: grading, dropping, and one retry."""

from __future__ import annotations

import json

from insightflow.rag.corrective import CorrectiveRetriever
from insightflow.rag.models import RetrievalQuery
from tests.fakes import ScriptedChatProvider, StaticRetriever, make_candidate


def grade(*relevant: int, query: str = "") -> str:
    return json.dumps({"relevant": list(relevant), "query": query})


QUESTION = RetrievalQuery(text="When does the term start?")


async def test_irrelevant_passages_are_dropped_and_ranks_renumbered() -> None:
    inner = StaticRetriever(
        [make_candidate("a", rank=1), make_candidate("b", rank=2), make_candidate("c", rank=3)]
    )
    chat = ScriptedChatProvider(grade=grade(2, 3, 9))

    results = await CorrectiveRetriever(inner, chat).retrieve(QUESTION)

    assert [(item.chunk.chunk_id, item.rank) for item in results] == [("b", 1), ("c", 2)]
    assert len(inner.queries) == 1
    prompt = chat.calls_for("grade")[0][1]["content"]
    assert prompt.startswith("Question: When does the term start?")
    assert "[2] PDF/overview.pdf" in prompt


async def test_no_relevant_passage_retries_with_the_graders_query() -> None:
    inner = StaticRetriever(
        [make_candidate("a")],
        by_query={"academic calendar semester dates": [make_candidate("x"), make_candidate("y")]},
    )
    chat = ScriptedChatProvider(
        grade=[grade(query="academic calendar semester dates"), grade(2)]
    )

    results = await CorrectiveRetriever(inner, chat).retrieve(QUESTION)

    assert [query.text for query in inner.queries] == [
        "When does the term start?",
        "academic calendar semester dates",
    ]
    assert [item.chunk.chunk_id for item in results] == ["y"]
    # The retry is graded against the original question.
    assert chat.calls_for("grade")[1][1]["content"].startswith(
        "Question: When does the term start?"
    )


async def test_there_is_at_most_one_retry() -> None:
    inner = StaticRetriever([make_candidate("a")])
    chat = ScriptedChatProvider(grade=grade(query="other words"))

    results = await CorrectiveRetriever(inner, chat).retrieve(QUESTION)

    assert results == []
    assert len(inner.queries) == 2
    assert len(chat.calls_for("grade")) == 2


async def test_no_relevant_passage_and_no_better_query_returns_nothing() -> None:
    inner = StaticRetriever([make_candidate("a")])
    same_query = grade(query=" when does the TERM start? ")

    for reply in (grade(), same_query):
        results = await CorrectiveRetriever(inner, ScriptedChatProvider(grade=reply)).retrieve(
            QUESTION
        )
        assert results == []
    assert len(inner.queries) == 2


async def test_unreadable_grades_keep_the_passages() -> None:
    inner = StaticRetriever(
        [make_candidate("a")], by_query={"better": [make_candidate("b")]}
    )

    first = await CorrectiveRetriever(inner, ScriptedChatProvider(grade="??")).retrieve(QUESTION)
    second = await CorrectiveRetriever(
        inner, ScriptedChatProvider(grade=[grade(query="better"), "??"])
    ).retrieve(QUESTION)

    assert [item.chunk.chunk_id for item in first] == ["a"]
    assert [item.chunk.chunk_id for item in second] == ["b"]


async def test_empty_retrieval_makes_no_model_call() -> None:
    chat = ScriptedChatProvider()

    assert await CorrectiveRetriever(StaticRetriever([]), chat).retrieve(QUESTION) == []
    assert chat.calls == []
