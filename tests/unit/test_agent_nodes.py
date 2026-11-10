"""Unit tests for individual workflow nodes and their helpers, without a graph."""

from datetime import UTC, datetime
from typing import Any

import pytest

from insightflow.agents.document_search import LOCAL_SERVER, DocumentSearch
from insightflow.agents.models import (
    SEARCH_TOOL,
    AgentAction,
    ConversationTurn,
    Observation,
    VerificationResult,
)
from insightflow.agents.nodes.act import make_act_node
from insightflow.agents.nodes.agent import make_agent_node
from insightflow.agents.nodes.answer import DIRECT_SYSTEM_PROMPT, make_answer_node
from insightflow.agents.nodes.finalize import finalize_node
from insightflow.agents.nodes.retry import retry_node
from insightflow.agents.nodes.verify import (
    NO_CONTEXT_ISSUE,
    check_citations,
    cited_passages,
    make_verify_node,
)
from insightflow.agents.observations import has_evidence, number_sources, render_observations
from insightflow.agents.state import AgentState, history_messages, new_turn_input
from insightflow.rag.models import ContextBudget
from insightflow.rag.retrieval import BudgetedContextAssembler
from insightflow.tools.models import ToolResult, WebSource
from insightflow.tools.registry import ToolRegistration, ToolSet
from tests.fakes import (
    ANSWER,
    FakeToolClient,
    ScriptedChatProvider,
    StaticRetriever,
    action,
    make_candidate,
    search,
    word_count,
)

HISTORY = [
    ConversationTurn(query="What is InsightFlow?", answer="A RAG platform [1]."),
    ConversationTurn(query="Who builds it?", answer="I do not know."),
]


def turn(query: str = "What is InsightFlow?", **fields: object) -> AgentState:
    state = new_turn_input(query)
    state.update(fields)  # type: ignore[typeddict-item]
    return state


def tool_result(
    tool: str = "calculate",
    *,
    status: Any = "ok",
    error: Any = "tool_failed",
    output: str = "4",
    web: list[WebSource] | None = None,
    arguments: dict[str, Any] | None = None,
) -> ToolResult:
    return ToolResult(
        call_id="c",
        server=LOCAL_SERVER if tool == SEARCH_TOOL else "tools",
        tool=tool,
        arguments=arguments or {},
        status=status,
        error=None if status == "ok" else error,
        output=output,
        web_sources=web or [],
        started_at=datetime.now(UTC),
        duration_ms=1.0,
    )


def searched(*chunk_ids: str, attempt: int = 1, query: str = "q") -> Observation:
    return Observation(
        attempt=attempt,
        result=tool_result(SEARCH_TOOL, output="", arguments={"query": query}),
        passages=[make_candidate(chunk_id, rank=n) for n, chunk_id in enumerate(chunk_ids, 1)],
    )


def document_search(retriever: StaticRetriever | None = None) -> DocumentSearch:
    return DocumentSearch(
        retriever or StaticRetriever(),
        BudgetedContextAssembler(word_count),
        top_k=4,
        budget=ContextBudget(max_tokens=100, max_chunks=2),
    )


# --- citations ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("No citation.", set()),
        ("One [1].", {1}),
        ("Grouped [1, 3] and [2].", {1, 2, 3}),
        ("Not a citation [a] or [].", set()),
    ],
)
def test_cited_passages(answer: str, expected: set[int]) -> None:
    assert cited_passages(answer) == expected


@pytest.mark.parametrize(
    ("answer", "sources", "issues"),
    [
        ("Supported [1][2].", 2, []),
        ("  ", 2, ["The answer is empty."]),
        ("No citation.", 2, ["The answer cites no source."]),
        ("Wrong [0] and [3].", 2, ["The answer cites sources that do not exist: [0], [3]."]),
    ],
)
def test_check_citations(answer: str, sources: int, issues: list[str]) -> None:
    assert check_citations(answer, sources) == issues


# --- observations ------------------------------------------------------------------------


def test_sources_are_numbered_once_across_steps() -> None:
    page = WebSource(title="Page", url="https://a.example/", content="text")
    observations = [
        searched("a", "b"),
        Observation(attempt=1, result=tool_result("web", web=[page, page])),
        searched("b", "c"),
    ]

    numbered = number_sources(observations)

    keys = [
        source.passage.chunk.chunk_id if source.passage else source.page.url  # type: ignore[union-attr]
        for source in numbered
    ]
    assert keys == ["a", "b", "https://a.example/", "c"]
    assert [source.number for source in numbered] == [1, 2, 3, 4]
    rendered = render_observations(observations)
    assert "[4] PDF/overview.pdf (page 2)" in rendered
    assert "Already shown: [2]" in rendered


def test_render_observations_describes_each_outcome() -> None:
    observations = [
        searched(query="nothing"),
        Observation(attempt=1, result=tool_result(status="error", output="bad input")),
        Observation(attempt=1, result=tool_result(output="42", arguments={"expression": "6*7"})),
    ]

    rendered = render_observations(observations)

    assert 'Step 1: search_documents {"query": "nothing"} found no sources.' in rendered
    assert "Step 2: calculate {} failed (tool_failed): bad input" in rendered
    assert 'Step 3: calculate {"expression": "6*7"} returned: 42' in rendered
    assert render_observations([]) == "No observations yet."


def test_has_evidence() -> None:
    assert not has_evidence([])
    assert not has_evidence([searched()])
    for error in ("timeout", "unavailable", "unknown_tool", "invalid_arguments"):
        failed = tool_result(status="error", error=error)
        assert not has_evidence([Observation(attempt=1, result=failed)])
    assert has_evidence([searched("a")])
    assert has_evidence([Observation(attempt=1, result=tool_result())])
    reported = tool_result(status="error", error="tool_failed")
    assert has_evidence([Observation(attempt=1, result=reported)])


# --- agent -------------------------------------------------------------------------------


async def test_agent_sends_history_catalog_question_and_budget() -> None:
    chat = ScriptedChatProvider(agent=search("Who builds InsightFlow?"))
    tools = ToolSet([ToolRegistration(FakeToolClient("fake", ["lookup"]))])
    node = make_agent_node(chat, tools, max_steps=3, history_turns=1)

    update = await node(turn("Who builds it?", history=HISTORY))

    assert update["pending_action"] == AgentAction(
        thought="Use search_documents.",
        action=SEARCH_TOOL,
        arguments={"query": "Who builds InsightFlow?"},
    )
    prompt = chat.calls_for("agent")[0][1]["content"]
    assert prompt.startswith(
        "Conversation so far:\nUser: Who builds it?\nAssistant: I do not know."
    )
    assert "What is InsightFlow?" not in prompt  # only the last turn is sent
    assert "- search_documents: Search the indexed documents" in prompt
    assert "- lookup: The lookup tool." in prompt
    assert '"maxLength": 500' in prompt
    assert "Question: Who builds it?" in prompt
    assert prompt.endswith("Observations so far:\nNo observations yet.\n\nSteps left: 3")


@pytest.mark.parametrize("reply", ['{"action": "Answer"}', '{"action": " answer "}', ANSWER])
async def test_agent_answer_action_is_normalized(reply: str) -> None:
    chat = ScriptedChatProvider(agent=reply)
    node = make_agent_node(chat, ToolSet(), max_steps=2, history_turns=0)

    update = await node(turn())

    pending = update["pending_action"]
    assert pending is not None and pending.is_answer


@pytest.mark.parametrize("reply", ["not json", '{"action": ""}', '{"thought": "no action"}'])
async def test_unusable_agent_output_searches_first_then_answers(reply: str) -> None:
    chat = ScriptedChatProvider(agent=reply)
    node = make_agent_node(chat, ToolSet(), max_steps=4, history_turns=0)

    first = await node(turn())
    later = await node(turn(observations=[searched("a")]))

    assert first["pending_action"] == AgentAction(
        thought="Search the documents for the question.",
        action=SEARCH_TOOL,
        arguments={"query": "What is InsightFlow?"},
    )
    assert later["pending_action"] is not None and later["pending_action"].is_answer


async def test_agent_answers_without_a_model_call_when_the_budget_is_used() -> None:
    chat = ScriptedChatProvider()
    node = make_agent_node(chat, ToolSet(), max_steps=2, history_turns=0)

    update = await node(turn(observations=[searched("a"), searched("b")]))

    assert update["pending_action"] is not None and update["pending_action"].is_answer
    assert chat.calls == []


async def test_the_budget_counts_only_the_current_attempt() -> None:
    chat = ScriptedChatProvider(agent=search("new"))
    node = make_agent_node(chat, ToolSet(), max_steps=2, history_turns=0)

    update = await node(turn(attempt=2, observations=[searched("a"), searched("b")]))

    assert update["pending_action"] is not None
    assert update["pending_action"].action == SEARCH_TOOL
    assert "Steps left: 2" in chat.calls_for("agent")[0][1]["content"]


async def test_agent_shows_verification_feedback() -> None:
    chat = ScriptedChatProvider(agent=ANSWER)
    node = make_agent_node(chat, ToolSet(), max_steps=2, history_turns=0)

    await node(turn(feedback=["made-up fact"], attempt=2))

    prompt = chat.calls_for("agent")[0][1]["content"]
    assert "Your previous answer failed verification:\n- made-up fact" in prompt


def test_agent_rejects_a_zero_budget() -> None:
    with pytest.raises(ValueError):
        make_agent_node(ScriptedChatProvider(), ToolSet(), max_steps=0, history_turns=0)


async def test_a_different_tool_or_argument_is_not_a_repeat() -> None:
    chat = ScriptedChatProvider(agent=[search("other"), action("lookup", query="q")])
    node = make_agent_node(chat, ToolSet(), max_steps=4, history_turns=0)
    state = turn(observations=[searched("a", query="q")])

    first = await node(state)
    second = await node(state)

    assert first["pending_action"] is not None and first["pending_action"].action == SEARCH_TOOL
    assert second["pending_action"] is not None and second["pending_action"].action == "lookup"


# --- act ---------------------------------------------------------------------------------


async def test_act_searches_documents_and_records_the_passages() -> None:
    retriever = StaticRetriever()
    node = make_act_node(document_search(retriever), ToolSet())
    pending = AgentAction(thought="look", action=SEARCH_TOOL, arguments={"query": " fees "})

    update = await node(turn(pending_action=pending, attempt=2))

    [observation] = update["observations"]
    assert [query.text for query in retriever.queries] == ["fees"]
    assert observation.attempt == 2
    assert observation.thought == "look"
    assert observation.result.server == LOCAL_SERVER
    assert observation.result.output == "1 passages"
    assert [passage.chunk.chunk_id for passage in observation.passages] == ["chunk-1"]
    assert update["pending_action"] is None


@pytest.mark.parametrize("arguments", [{}, {"query": "  "}, {"query": "x", "top_k": 5}])
async def test_act_rejects_invalid_search_arguments_without_searching(
    arguments: dict[str, Any],
) -> None:
    retriever = StaticRetriever()
    node = make_act_node(document_search(retriever), ToolSet())
    pending = AgentAction(action=SEARCH_TOOL, arguments=arguments)

    update = await node(turn(pending_action=pending))

    assert update["observations"][0].result.error == "invalid_arguments"
    assert retriever.queries == []


async def test_act_sends_other_actions_to_the_tool_set() -> None:
    client = FakeToolClient("fake", ["lookup"])
    tools = ToolSet([ToolRegistration(client)])
    await tools.list_tools()
    node = make_act_node(document_search(), tools)

    update = await node(turn(pending_action=AgentAction(action="lookup", arguments={"query": "x"})))

    assert client.calls == [("lookup", {"query": "x"})]
    assert update["observations"][0].passages == []


async def test_act_needs_a_tool_action() -> None:
    node = make_act_node(document_search(), ToolSet())

    with pytest.raises(ValueError):
        await node(turn(pending_action=AgentAction(action="answer")))


# --- answer ------------------------------------------------------------------------------


async def test_direct_answer_uses_history_and_no_observations() -> None:
    chat = ScriptedChatProvider()
    node = make_answer_node(chat, history_turns=1)

    update = await node(turn("thanks", history=HISTORY))

    assert update == {"answer": "Hello! Ask me about the indexed documents."}
    assert chat.calls_for("direct")[0] == [
        {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
        {"role": "user", "content": "Who builds it?"},
        {"role": "assistant", "content": "I do not know."},
        {"role": "user", "content": "thanks"},
    ]


def test_history_messages_respects_the_turn_limit() -> None:
    state = turn(history=HISTORY)

    assert history_messages(state, 0) == []
    assert len(history_messages(state, 1)) == 2
    assert len(history_messages(state, 10)) == 4


# --- verify ------------------------------------------------------------------------------


async def test_verify_skips_when_nothing_was_found() -> None:
    chat = ScriptedChatProvider()

    update = await make_verify_node(chat)(turn(observations=[searched()], answer="x"))

    assert update["verification"] == VerificationResult(
        status="skipped", method="deterministic", issues=[NO_CONTEXT_ISSUE]
    )
    assert chat.calls == []


async def test_verify_skips_tool_only_answers_without_an_issue() -> None:
    state = turn(observations=[Observation(attempt=1, result=tool_result())], answer="4")

    update = await make_verify_node(ScriptedChatProvider())(state)

    assert update["verification"] == VerificationResult(status="skipped", method="deterministic")


async def test_verify_fails_bad_citations_without_calling_the_judge() -> None:
    chat = ScriptedChatProvider()

    update = await make_verify_node(chat)(turn(observations=[searched("a")], answer="Claim [2]."))

    verification = update["verification"]
    assert verification is not None
    assert (verification.status, verification.method) == ("failed", "deterministic")
    assert chat.calls == []


async def test_verify_counts_sources_from_every_step() -> None:
    state = turn(observations=[searched("a"), searched("b")], answer="Claims [1] and [2].")

    update = await make_verify_node(None)(state)

    assert update["verification"] == VerificationResult(status="passed", method="deterministic")


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (
            '{"grounded": true, "unsupported_claims": []}',
            VerificationResult(status="passed", method="llm"),
        ),
        (
            '{"grounded": false, "unsupported_claims": ["founded in 1990", " "]}',
            VerificationResult(status="failed", method="llm", issues=["founded in 1990"]),
        ),
        (
            '{"grounded": false}',
            VerificationResult(
                status="failed", method="llm", issues=["The judge found unsupported claims."]
            ),
        ),
        (
            "maybe",
            VerificationResult(
                status="unchecked", method="llm", issues=["The judge response could not be read."]
            ),
        ),
    ],
)
async def test_verify_uses_the_llm_judge(reply: str, expected: VerificationResult) -> None:
    chat = ScriptedChatProvider(judge=reply)
    state = turn(observations=[searched("a")], answer="Claim [1].")

    update = await make_verify_node(chat)(state)

    assert update["verification"] == expected
    prompt = chat.calls_for("judge")[0][1]["content"]
    assert prompt.startswith("Observations:\nStep 1: search_documents")
    assert "Question: What is InsightFlow?" in prompt
    assert prompt.endswith("Answer: Claim [1].")


# --- retry and finalize ------------------------------------------------------------------


async def test_retry_starts_the_next_attempt_with_the_issues() -> None:
    failed = VerificationResult(status="failed", method="llm", issues=["unsupported claim"])

    update = await retry_node(turn(verification=failed, attempt=1, answer="old"))

    assert update == {
        "attempt": 2,
        "feedback": ["unsupported claim"],
        "answer": "",
        "verification": None,
    }


async def test_retry_without_issues_gives_a_generic_reason() -> None:
    update = await retry_node(turn(verification=None, attempt=2))

    assert update["feedback"] == ["The previous answer was not supported by the observations."]


async def test_finalize_records_the_turn_and_marks_direct_answers_skipped() -> None:
    update = await finalize_node(turn("hi", answer="Hello!"))

    assert update == {
        "history": [ConversationTurn(query="hi", answer="Hello!")],
        "verification": VerificationResult(status="skipped", method="none"),
    }


async def test_finalize_keeps_an_existing_verification() -> None:
    verification = VerificationResult(status="passed", method="llm")

    update = await finalize_node(turn(answer="A [1].", verification=verification))

    assert "verification" not in update
