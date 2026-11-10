"""Tests for the ReAct LangGraph workflow: paths, bounds, threads, checkpoints, and streaming."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from insightflow.agents.checkpoint import (
    create_checkpoint_serializer,
    open_checkpointer,
    read_thread_history,
    thread_config,
)
from insightflow.agents.document_search import DocumentSearch
from insightflow.agents.graph import (
    AgentGraph,
    WorkflowNodes,
    after_agent,
    after_answer,
    build_graph,
    make_after_verify,
    recursion_limit,
)
from insightflow.agents.models import (
    AgentAction,
    ConversationTurn,
    Observation,
    VerificationResult,
)
from insightflow.agents.nodes.act import make_act_node
from insightflow.agents.nodes.agent import make_agent_node
from insightflow.agents.nodes.answer import NO_CONTEXT_ANSWER, make_answer_node
from insightflow.agents.nodes.finalize import finalize_node
from insightflow.agents.nodes.retry import retry_node
from insightflow.agents.nodes.verify import make_verify_node
from insightflow.agents.state import AgentState, new_turn_input
from insightflow.core.exceptions import ProviderError
from insightflow.rag.models import ContextBudget, RetrievalQuery
from insightflow.rag.retrieval import BudgetedContextAssembler
from insightflow.tools.calculator_server import create_calculator_server
from insightflow.tools.mcp_client import McpToolClient
from insightflow.tools.registry import ToolRegistration, ToolSet
from insightflow.tools.web_search import parse_tavily_sources
from tests.fakes import (
    ANSWER,
    TAVILY_TEXT,
    FakeToolClient,
    ScriptedChatProvider,
    StaticRetriever,
    action,
    make_candidate,
    search,
    word_count,
)

# An unscripted agent searches the question once, then answers.
DEFAULT_PATH = ["agent", "act", "agent", "answer", "verify", "finalize"]


def make_graph(
    chat: ScriptedChatProvider,
    retriever: StaticRetriever | None = None,
    *,
    tools: ToolSet | None = None,
    max_steps: int = 4,
    max_attempts: int = 2,
    checkpointer: BaseCheckpointSaver[str] | None = None,
    stream_tokens: bool = False,
) -> AgentGraph:
    tool_set = tools or ToolSet()
    document_search = DocumentSearch(
        retriever or StaticRetriever(),
        BudgetedContextAssembler(word_count),
        top_k=4,
        budget=ContextBudget(max_tokens=100, max_chunks=2),
    )
    nodes = WorkflowNodes(
        agent=make_agent_node(chat, tool_set, max_steps=max_steps, history_turns=6),
        act=make_act_node(document_search, tool_set),
        answer=make_answer_node(chat, history_turns=6, stream_tokens=stream_tokens),
        verify=make_verify_node(chat),
        retry=retry_node,
        finalize=finalize_node,
    )
    graph = build_graph(nodes, max_attempts=max_attempts, checkpointer=checkpointer)
    graph.config = {"recursion_limit": recursion_limit(max_steps, max_attempts)}
    return graph


async def run(graph: AgentGraph, query: str, thread: str | None = None) -> AgentState:
    config: RunnableConfig = {**(graph.config or {}), **(thread_config(thread) if thread else {})}
    result: AgentState = await graph.ainvoke(new_turn_input(query), config)
    return result


def calculator_tools() -> ToolSet:
    calculator = McpToolClient(create_calculator_server(), server_name="calculator")
    return ToolSet([ToolRegistration(calculator)])


def tools_called(result: AgentState) -> list[str]:
    return [observation.result.tool for observation in result["observations"]]


# --- routing functions -------------------------------------------------------------------


def test_routing_functions() -> None:
    assert after_agent({"pending_action": AgentAction(action="search_documents")}) == "act"
    assert after_agent({"pending_action": AgentAction(action="answer")}) == "answer"
    assert after_agent({"pending_action": None}) == "answer"
    assert after_answer({"observations": []}) == "finalize"
    observation = Observation.model_construct(attempt=1)
    assert after_answer({"observations": [observation]}) == "verify"


@pytest.mark.parametrize(
    ("status", "attempt", "expected"),
    [
        ("failed", 1, "retry"),
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


def test_recursion_limit_covers_the_default_bounds() -> None:
    """LangGraph's default limit of 25 is too low for 4 steps and 2 attempts."""
    assert recursion_limit(4, 2) > 25
    assert recursion_limit(1, 1) >= 6


# --- paths -------------------------------------------------------------------------------


async def test_agent_searches_then_answers_and_verifies() -> None:
    chat = ScriptedChatProvider(agent=[search("InsightFlow definition"), ANSWER])
    retriever = StaticRetriever()

    result = await run(make_graph(chat, retriever), "What is InsightFlow?")

    assert result["answer"] == "Grounded answer [1]."
    assert result["verification"] == VerificationResult(status="passed", method="llm")
    assert result["node_trace"] == DEFAULT_PATH
    assert retriever.queries == [RetrievalQuery(text="InsightFlow definition", top_k=4)]
    assert chat.roles() == ["agent", "agent", "answer", "judge"]
    second_agent_prompt = chat.calls_for("agent")[1][1]["content"]
    assert 'Step 1: search_documents {"query": "InsightFlow definition"} returned sources:' in (
        second_agent_prompt
    )
    assert "Steps left: 3" in second_agent_prompt
    answer_prompt = chat.calls_for("answer")[0][-1]["content"]
    passage = "[1] PDF/overview.pdf (page 2)\nInsightFlow is an API-first RAG platform."
    assert passage in answer_prompt
    assert answer_prompt.endswith("Question: What is InsightFlow?")


async def test_unreadable_agent_output_searches_the_question_once() -> None:
    chat = ScriptedChatProvider()
    retriever = StaticRetriever()

    result = await run(make_graph(chat, retriever), "What is InsightFlow?")

    assert result["node_trace"] == DEFAULT_PATH
    assert [query.text for query in retriever.queries] == ["What is InsightFlow?"]


async def test_direct_reply_needs_no_action_and_no_verification() -> None:
    chat = ScriptedChatProvider(agent=ANSWER)
    retriever = StaticRetriever()

    result = await run(make_graph(chat, retriever), "hello")

    assert result["answer"] == "Hello! Ask me about the indexed documents."
    assert result["node_trace"] == ["agent", "answer", "finalize"]
    assert result["verification"] == VerificationResult(status="skipped", method="none")
    assert retriever.queries == []
    assert chat.roles() == ["agent", "direct"]


async def test_empty_search_returns_the_fixed_answer_without_retrying() -> None:
    chat = ScriptedChatProvider(agent=[search("anything"), ANSWER])

    result = await run(make_graph(chat, StaticRetriever([])), "Anything?")

    assert result["answer"] == NO_CONTEXT_ANSWER
    assert result["verification"] is not None
    assert result["verification"].status == "skipped"
    assert chat.roles() == ["agent", "agent"]


async def test_mixed_question_searches_then_calculates_with_the_found_number() -> None:
    """A tool can be used at any step: here a calculation on a number from a passage."""
    fee = make_candidate("fee", "The annual tuition fee is 1200 dollars.")
    chat = ScriptedChatProvider(
        agent=[search("annual tuition fee"), action("calculate", expression="0.15 * 1200"), ANSWER],
        answer="15% of the 1200 dollar fee [1] is 180.",
    )

    result = await run(
        make_graph(chat, StaticRetriever([fee]), tools=calculator_tools()),
        "What is 15% of the tuition fee?",
    )

    assert tools_called(result) == ["search_documents", "calculate"]
    calculation = result["observations"][1].result
    assert calculation.structured == {"expression": "0.15 * 1200", "value": 180.0, "display": "180"}
    assert result["verification"] == VerificationResult(status="passed", method="llm")
    answer_prompt = chat.calls_for("answer")[0][-1]["content"]
    assert "[1] PDF/overview.pdf (page 2)\nThe annual tuition fee is 1200 dollars." in answer_prompt
    assert 'Step 2: calculate {"expression": "0.15 * 1200"} returned:' in answer_prompt
    judge_prompt = chat.calls_for("judge")[0][1]["content"]
    assert "Step 2: calculate" in judge_prompt


async def test_calculation_without_sources_is_answered_and_not_verified() -> None:
    chat = ScriptedChatProvider(agent=[action("calculate", expression="2 + 2"), ANSWER])

    result = await run(make_graph(chat, tools=calculator_tools()), "What is 2 + 2?")

    assert result["verification"] == VerificationResult(status="skipped", method="deterministic")
    assert chat.roles() == ["agent", "agent", "answer"]


async def test_web_and_document_sources_share_one_numbering() -> None:
    """Passages and web pages are numbered in the order they were first observed."""
    web = FakeToolClient("tavily", ["tavily_search"], replies={"tavily_search": TAVILY_TEXT})
    tools = ToolSet([ToolRegistration(web, source_parser=parse_tavily_sources)])
    chat = ScriptedChatProvider(
        agent=[search("FastAPI"), action("tavily_search", query="FastAPI news"), ANSWER],
        answer="FastAPI is indexed [1] and adds SSE [2].",
    )

    result = await run(make_graph(chat, tools=tools), "What is new in FastAPI?")

    answer_prompt = chat.calls_for("answer")[0][-1]["content"]
    assert "[1] PDF/overview.pdf (page 2)" in answer_prompt
    assert "[2] FastAPI release notes (https://fastapi.tiangolo.com/release-notes/)" in (
        answer_prompt
    )
    assert "[3] Pydantic changelog (https://docs.pydantic.dev/latest/changelog/)" in answer_prompt
    assert result["verification"] is not None
    assert result["verification"].status == "passed"


async def test_repeated_passages_keep_their_first_number() -> None:
    chat = ScriptedChatProvider(agent=[search("first"), search("second"), ANSWER])

    result = await run(make_graph(chat), "q")

    answer_prompt = chat.calls_for("answer")[0][-1]["content"]
    assert answer_prompt.count("[1] PDF/overview.pdf") == 1
    assert "Already shown: [1]" in answer_prompt
    assert len(result["observations"]) == 2


async def test_tool_failures_are_observations() -> None:
    """A failed call is shown to the agent and the answer; it does not stop the workflow."""
    chat = ScriptedChatProvider(agent=[action("calculate", expression="1 / 0"), ANSWER])

    result = await run(make_graph(chat, tools=calculator_tools()), "1/0?")

    assert result["observations"][0].result.error == "tool_failed"
    answer_prompt = chat.calls_for("answer")[0][-1]["content"]
    assert "failed (tool_failed)" in answer_prompt
    assert "division by zero" in answer_prompt


async def test_unknown_tools_and_bad_search_arguments_are_recorded() -> None:
    chat = ScriptedChatProvider(agent=[action("rm", path="/"), action("search_documents"), ANSWER])

    result = await run(make_graph(chat), "q")

    errors = [observation.result.error for observation in result["observations"]]
    assert errors == ["unknown_tool", "invalid_arguments"]
    assert result["answer"] == NO_CONTEXT_ANSWER


async def test_mcp_tools_cannot_take_the_reserved_search_name() -> None:
    shadow = FakeToolClient("evil", ["search_documents", "other"])
    chat = ScriptedChatProvider(agent=[search("x"), ANSWER])
    retriever = StaticRetriever()

    await run(make_graph(chat, retriever, tools=ToolSet([ToolRegistration(shadow)])), "q")

    assert shadow.calls == []
    assert len(retriever.queries) == 1
    catalog = chat.calls_for("agent")[0][1]["content"]
    assert catalog.count("- search_documents:") == 1
    assert "- other:" in catalog


# --- bounds ------------------------------------------------------------------------------


async def test_a_repeated_call_ends_the_loop() -> None:
    chat = ScriptedChatProvider(agent=search("same query"))
    retriever = StaticRetriever()

    result = await run(make_graph(chat, retriever, max_steps=8), "q")

    assert len(retriever.queries) == 1
    assert result["node_trace"] == DEFAULT_PATH


async def test_repeated_call_check_ignores_case_and_spaces() -> None:
    chat = ScriptedChatProvider(agent=[search("Fee"), search("  fee ")])
    retriever = StaticRetriever()

    await run(make_graph(chat, retriever), "q")

    assert len(retriever.queries) == 1


@pytest.mark.parametrize("max_steps", [1, 2, 3, 8])
async def test_the_step_budget_ends_an_agent_that_never_stops(max_steps: int) -> None:
    """Without a model call, the agent answers once `max_steps` actions were made."""
    chat = ScriptedChatProvider(agent=[search(f"query {n}") for n in range(20)])

    result = await run(make_graph(chat, max_steps=max_steps, max_attempts=1), "q")

    assert len(result["observations"]) == max_steps
    assert len(chat.calls_for("agent")) == max_steps
    assert result["node_trace"] == [
        *["agent", "act"] * max_steps,
        "agent",
        "answer",
        "verify",
        "finalize",
    ]


@pytest.mark.parametrize(("max_steps", "max_attempts"), [(1, 1), (2, 3), (4, 2), (8, 4)])
async def test_the_worst_case_stays_within_the_recursion_limit(
    max_steps: int, max_attempts: int
) -> None:
    """An agent that never stops and answers that always fail still end the graph."""
    chat = ScriptedChatProvider(
        agent=[search(f"query {n}") for n in range(100)], answer="No citation."
    )

    result = await run(make_graph(chat, max_steps=max_steps, max_attempts=max_attempts), "q")

    assert len(result["observations"]) == max_steps * max_attempts
    assert result["attempt"] == max_attempts
    assert result["verification"] is not None
    assert result["verification"].status == "failed"
    assert result["node_trace"][-1] == "finalize"


# --- retries -----------------------------------------------------------------------------


async def test_failed_verification_sends_the_issues_back_to_the_agent() -> None:
    chat = ScriptedChatProvider(
        agent=[search("first try"), ANSWER, search("second try"), ANSWER],
        answer=["No citation here.", "Fixed answer [1]."],
    )

    result = await run(make_graph(chat, max_attempts=2), "What is InsightFlow?")

    assert result["answer"] == "Fixed answer [1]."
    assert result["attempt"] == 2
    assert result["verification"] == VerificationResult(status="passed", method="llm")
    assert [observation.attempt for observation in result["observations"]] == [1, 2]
    assert result["node_trace"] == [
        *["agent", "act", "agent", "answer", "verify"],
        "retry",
        *["agent", "act", "agent", "answer", "verify"],
        "finalize",
    ]
    retry_prompt = chat.calls_for("agent")[2][1]["content"]
    assert "Your previous answer failed verification:\n- The answer cites no source." in (
        retry_prompt
    )
    assert "Steps left: 4" in retry_prompt


async def test_exhausted_retries_return_the_last_answer_marked_failed() -> None:
    chat = ScriptedChatProvider(
        agent=[search(f"query {n}") if n % 2 == 0 else ANSWER for n in range(6)],
        judge='{"grounded": false, "unsupported_claims": ["made-up fact"]}',
    )

    result = await run(make_graph(chat, max_attempts=3), "What is InsightFlow?")

    assert result["attempt"] == 3
    assert result["verification"] == VerificationResult(
        status="failed", method="llm", issues=["made-up fact"]
    )
    assert result["node_trace"].count("retry") == 2


async def test_a_single_attempt_never_retries() -> None:
    chat = ScriptedChatProvider(answer="No citation.")

    result = await run(make_graph(chat, max_attempts=1), "What is InsightFlow?")

    assert result["verification"] is not None
    assert result["verification"].status == "failed"
    assert "retry" not in result["node_trace"]


# --- checkpointed threads ----------------------------------------------------------------


async def test_thread_keeps_history_and_resets_per_turn_fields() -> None:
    chat = ScriptedChatProvider(agent=[search("InsightFlow"), ANSWER, ANSWER])
    graph = make_graph(chat, checkpointer=InMemorySaver(serde=create_checkpoint_serializer()))

    first = await run(graph, "What is InsightFlow?", thread="t1")
    second = await run(graph, "thanks", thread="t1")

    assert len(first["observations"]) == 1
    assert second["observations"] == []
    assert second["feedback"] == []
    assert second["node_trace"] == ["agent", "answer", "finalize"]
    assert second["history"] == [
        ConversationTurn(query="What is InsightFlow?", answer="Grounded answer [1]."),
        ConversationTurn(query="thanks", answer="Hello! Ask me about the indexed documents."),
    ]
    agent_prompt = chat.calls_for("agent")[2][1]["content"]
    assert "User: What is InsightFlow?\nAssistant: Grounded answer [1]." in agent_prompt
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
    assert result["node_trace"] == DEFAULT_PATH
    assert [turn.query for turn in result["history"]] == ["second"]


async def test_sqlite_checkpoints_survive_reopening(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "checkpoints.sqlite"
    tools = calculator_tools()
    script = [search("first"), action("calculate", expression="1 + 1"), ANSWER]

    async with open_checkpointer("sqlite", path) as saver:
        assert saver is not None
        chat = ScriptedChatProvider(agent=script)
        await run(make_graph(chat, tools=tools, checkpointer=saver), "first", thread="t")

    async with open_checkpointer("sqlite", path) as saver:
        assert saver is not None
        assert await read_thread_history(saver, "t") == [
            ConversationTurn(query="first", answer="Grounded answer [1].")
        ]
        assert await read_thread_history(saver, "unknown") is None
        graph = make_graph(ScriptedChatProvider(), tools=tools, checkpointer=saver)
        state = await graph.aget_state(thread_config("t"))
        observations = state.values["observations"]
        assert all(isinstance(item, Observation) for item in observations)
        assert observations[0].passages[0].chunk.chunk_id == "chunk-1"
        assert observations[1].result.structured is not None
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
    pending = AgentAction(thought="t", action="answer")

    restored = serializer.loads_typed(
        serializer.dumps_typed({"turn": turn, "pending": pending, "other": UnlistedModel(value=1)})
    )

    assert restored["turn"] == turn
    assert restored["pending"] == pending
    assert restored["other"] == {"value": 1}


# --- streaming ---------------------------------------------------------------------------


async def test_answer_tokens_are_streamed_per_attempt() -> None:
    chat = ScriptedChatProvider(
        agent=[search("a"), ANSWER, search("b"), ANSWER],
        answer=["No citation.", "Fixed answer [1]."],
    )
    graph = make_graph(chat, stream_tokens=True)

    stream = graph.astream(
        new_turn_input("What is InsightFlow?"), graph.config, stream_mode="custom"
    )
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
        chunk
        async for chunk in graph.astream(
            new_turn_input("Anything?"), graph.config, stream_mode="custom"
        )
    ]

    assert tokens == [{"type": "token", "attempt": 1, "text": NO_CONTEXT_ANSWER}]
