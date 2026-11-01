"""Tests for the routed, verified LangGraph workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from insightflow.agents.checkpoint import (
    create_checkpoint_serializer,
    open_checkpointer,
    read_thread_history,
    thread_config,
)
from insightflow.agents.graph import (
    AgentGraph,
    WorkflowNodes,
    after_answer,
    after_route,
    build_graph,
    make_after_verify,
)
from insightflow.agents.models import ConversationTurn, VerificationResult
from insightflow.agents.nodes.answer import NO_CONTEXT_ANSWER, make_answer_node
from insightflow.agents.nodes.finalize import finalize_node
from insightflow.agents.nodes.retrieve import make_retrieve_node
from insightflow.agents.nodes.rewrite import make_rewrite_node
from insightflow.agents.nodes.route import make_route_node
from insightflow.agents.nodes.verify import make_verify_node
from insightflow.agents.state import AgentState, new_turn_input
from insightflow.core.exceptions import ProviderError
from insightflow.rag.models import AssembledContext, ContextBudget, RetrievalQuery
from insightflow.rag.retrieval import BudgetedContextAssembler
from tests.fakes import ScriptedChatProvider, StaticRetriever, word_count

FULL_PATH = ["route", "retrieve", "answer", "verify", "finalize"]


def make_graph(
    chat: ScriptedChatProvider,
    retriever: StaticRetriever | None = None,
    *,
    max_attempts: int = 2,
    checkpointer: BaseCheckpointSaver[str] | None = None,
    stream_tokens: bool = False,
) -> AgentGraph:
    nodes = WorkflowNodes(
        route=make_route_node(chat, history_turns=6),
        retrieve=make_retrieve_node(
            retriever or StaticRetriever(),
            BudgetedContextAssembler(word_count),
            top_k=4,
            budget=ContextBudget(max_tokens=100, max_chunks=2),
        ),
        answer=make_answer_node(chat, history_turns=6, stream_tokens=stream_tokens),
        verify=make_verify_node(chat),
        rewrite=make_rewrite_node(chat),
        finalize=finalize_node,
    )
    return build_graph(nodes, max_attempts=max_attempts, checkpointer=checkpointer)


async def run(graph: AgentGraph, query: str, thread: str | None = None) -> AgentState:
    config = thread_config(thread) if thread else None
    result: AgentState = await graph.ainvoke(new_turn_input(query), config)
    return result


# --- routing functions -------------------------------------------------------------------


def test_routing_functions() -> None:
    assert after_route({"route": "direct"}) == "answer"
    assert after_route({"route": "retrieve"}) == "retrieve"
    assert after_route({}) == "retrieve"
    assert after_answer({"route": "direct"}) == "finalize"
    assert after_answer({"route": "retrieve"}) == "verify"


@pytest.mark.parametrize(
    ("status", "attempt", "expected"),
    [
        ("failed", 1, "rewrite"),
        ("failed", 2, "finalize"),
        ("passed", 1, "finalize"),
        ("unchecked", 1, "finalize"),
        ("skipped", 1, "finalize"),
    ],
)
def test_after_verify_retries_failures_within_the_bound(
    status: Any, attempt: int, expected: str
) -> None:
    state: AgentState = {
        "verification": VerificationResult(status=status, method="llm"),
        "attempt": attempt,
    }
    assert make_after_verify(2)(state) == expected


def test_after_verify_rejects_a_zero_bound() -> None:
    with pytest.raises(ValueError):
        make_after_verify(0)


# --- paths -------------------------------------------------------------------------------


async def test_retrieve_path_answers_from_context_and_verifies() -> None:
    chat = ScriptedChatProvider(
        route='{"route": "retrieve", "standalone_query": "InsightFlow definition"}'
    )
    retriever = StaticRetriever()

    result = await run(make_graph(chat, retriever), "What is InsightFlow?")

    assert result["answer"] == "Grounded answer [1]."
    assert result["verification"] == VerificationResult(status="passed", method="llm")
    assert result["node_trace"] == FULL_PATH
    assert result["retrieval_queries"] == ["InsightFlow definition"]
    assert retriever.queries == [RetrievalQuery(text="InsightFlow definition", top_k=4)]
    assert chat.roles() == ["route", "answer", "judge"]
    answer_prompt = chat.calls_for("answer")[0][-1]["content"]
    passage = "[1] PDF/overview.pdf (page 2)\nInsightFlow is an API-first RAG platform."
    assert passage in answer_prompt
    assert answer_prompt.endswith("Question: What is InsightFlow?")


async def test_direct_path_skips_retrieval_and_verification() -> None:
    chat = ScriptedChatProvider(route='{"route": "direct", "standalone_query": "hello"}')
    retriever = StaticRetriever()

    result = await run(make_graph(chat, retriever), "hello")

    assert result["answer"] == "Hello! Ask me about the indexed documents."
    assert result["node_trace"] == ["route", "answer", "finalize"]
    assert result["verification"] == VerificationResult(status="skipped", method="none")
    assert result["context"] is None
    assert retriever.queries == []
    assert chat.roles() == ["route", "direct"]


async def test_empty_retrieval_returns_the_fixed_answer_without_retrying() -> None:
    chat = ScriptedChatProvider()

    result = await run(make_graph(chat, StaticRetriever([])), "Anything?")

    assert result["answer"] == NO_CONTEXT_ANSWER
    assert result["verification"] is not None
    assert result["verification"].status == "skipped"
    assert result["node_trace"] == FULL_PATH
    assert chat.roles() == ["route"]


async def test_failed_verification_rewrites_the_query_and_retries() -> None:
    chat = ScriptedChatProvider(answer=["No citation here.", "Fixed answer [1]."])
    retriever = StaticRetriever()

    result = await run(make_graph(chat, retriever, max_attempts=2), "What is InsightFlow?")

    assert result["answer"] == "Fixed answer [1]."
    assert result["attempt"] == 2
    assert result["verification"] == VerificationResult(status="passed", method="llm")
    assert result["retrieval_queries"] == ["What is InsightFlow?", "rewritten query"]
    assert result["node_trace"] == [
        "route", "retrieve", "answer", "verify", "rewrite", "retrieve", "answer", "verify",
        "finalize",
    ]
    assert chat.roles() == ["route", "answer", "rewrite", "answer", "judge"]


async def test_exhausted_retries_return_the_last_answer_marked_failed() -> None:
    chat = ScriptedChatProvider(
        judge='{"grounded": false, "unsupported_claims": ["made-up fact"]}'
    )

    result = await run(make_graph(chat, max_attempts=3), "What is InsightFlow?")

    assert result["attempt"] == 3
    assert result["verification"] == VerificationResult(
        status="failed", method="llm", issues=["made-up fact"]
    )
    assert chat.roles().count("rewrite") == 2
    assert result["node_trace"][-1] == "finalize"


async def test_a_single_attempt_never_rewrites() -> None:
    chat = ScriptedChatProvider(answer="No citation.")

    result = await run(make_graph(chat, max_attempts=1), "What is InsightFlow?")

    assert result["verification"] is not None
    assert result["verification"].status == "failed"
    assert "rewrite" not in chat.roles()


# --- checkpointed threads ----------------------------------------------------------------


async def test_thread_keeps_history_and_resets_per_turn_fields() -> None:
    chat = ScriptedChatProvider(
        route=[
            '{"route": "retrieve", "standalone_query": "What is InsightFlow?"}',
            '{"route": "direct", "standalone_query": "Thanks"}',
        ]
    )
    graph = make_graph(chat, checkpointer=InMemorySaver(serde=create_checkpoint_serializer()))

    first = await run(graph, "What is InsightFlow?", thread="t1")
    second = await run(graph, "thanks", thread="t1")

    assert first["route"] == "retrieve"
    assert second["route"] == "direct"
    assert second["context"] is None
    assert second["retrieval_queries"] == []
    assert second["node_trace"] == ["route", "answer", "finalize"]
    assert second["history"] == [
        ConversationTurn(query="What is InsightFlow?", answer="Grounded answer [1]."),
        ConversationTurn(query="thanks", answer="Hello! Ask me about the indexed documents."),
    ]
    router_prompt = chat.calls_for("route")[1][1]["content"]
    assert "User: What is InsightFlow?\nAssistant: Grounded answer [1]." in router_prompt
    direct_messages = chat.calls_for("direct")[0]
    assert direct_messages[1] == {"role": "user", "content": "What is InsightFlow?"}


async def test_threads_are_isolated() -> None:
    graph = make_graph(
        ScriptedChatProvider(), checkpointer=InMemorySaver(serde=create_checkpoint_serializer())
    )

    await run(graph, "first", thread="a")
    result = await run(graph, "second", thread="b")

    assert [turn.query for turn in result["history"]] == ["second"]


async def test_a_failed_turn_does_not_break_the_thread() -> None:
    chat = ScriptedChatProvider(answer=[ProviderError(), "Recovered [1]."])
    saver = InMemorySaver(serde=create_checkpoint_serializer())
    graph = make_graph(chat, checkpointer=saver)

    with pytest.raises(ProviderError):
        await run(graph, "first", thread="t")
    result = await run(graph, "second", thread="t")

    assert result["answer"] == "Recovered [1]."
    assert result["node_trace"] == FULL_PATH
    assert [turn.query for turn in result["history"]] == ["second"]


async def test_sqlite_checkpoints_survive_reopening(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "checkpoints.sqlite"

    async with open_checkpointer("sqlite", path) as saver:
        assert saver is not None
        await run(make_graph(ScriptedChatProvider(), checkpointer=saver), "first", thread="t")

    async with open_checkpointer("sqlite", path) as saver:
        assert saver is not None
        assert await read_thread_history(saver, "t") == [
            ConversationTurn(query="first", answer="Grounded answer [1].")
        ]
        assert await read_thread_history(saver, "unknown") is None
        graph = make_graph(ScriptedChatProvider(), checkpointer=saver)
        state = await graph.aget_state(thread_config("t"))
        assert isinstance(state.values["context"], AssembledContext)
        assert isinstance(state.values["verification"], VerificationResult)
        result = await run(graph, "second", thread="t")
        assert [turn.query for turn in result["history"]] == ["first", "second"]


async def test_open_checkpointer_backends(tmp_path: Path) -> None:
    async with open_checkpointer("none", tmp_path / "unused.sqlite") as saver:
        assert saver is None
    async with open_checkpointer("memory", tmp_path / "unused.sqlite") as saver:
        assert isinstance(saver, InMemorySaver)
    assert not (tmp_path / "unused.sqlite").exists()


class UnlistedModel(BaseModel):
    value: int


def test_checkpoint_serializer_restores_only_allowed_models() -> None:
    serializer = create_checkpoint_serializer()
    turn = ConversationTurn(query="q", answer="a")

    restored = serializer.loads_typed(
        serializer.dumps_typed({"turn": turn, "other": UnlistedModel(value=1)})
    )

    assert restored["turn"] == turn
    assert restored["other"] == {"value": 1}


# --- streaming ---------------------------------------------------------------------------


async def test_answer_tokens_are_streamed_per_attempt() -> None:
    chat = ScriptedChatProvider(answer=["No citation.", "Fixed answer [1]."])
    graph = make_graph(chat, stream_tokens=True)

    stream = graph.astream(new_turn_input("What is InsightFlow?"), stream_mode="custom")
    tokens = [chunk async for chunk in stream]

    assert tokens == [
        {"type": "token", "attempt": 1, "text": "No "},
        {"type": "token", "attempt": 1, "text": "citation."},
        {"type": "token", "attempt": 2, "text": "Fixed "},
        {"type": "token", "attempt": 2, "text": "answer "},
        {"type": "token", "attempt": 2, "text": "[1]."},
    ]
    assert chat.streamed == ["answer", "answer"]


async def test_fixed_no_context_answer_is_streamed_as_one_token() -> None:
    graph = make_graph(ScriptedChatProvider(), StaticRetriever([]), stream_tokens=True)

    tokens = [
        chunk async for chunk in graph.astream(new_turn_input("Anything?"), stream_mode="custom")
    ]

    assert tokens == [{"type": "token", "attempt": 1, "text": NO_CONTEXT_ANSWER}]
