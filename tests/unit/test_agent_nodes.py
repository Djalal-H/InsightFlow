"""Unit tests for individual workflow nodes, without a graph."""

import pytest

from insightflow.agents.models import ConversationTurn, VerificationResult
from insightflow.agents.nodes.answer import DIRECT_SYSTEM_PROMPT, make_answer_node
from insightflow.agents.nodes.finalize import finalize_node
from insightflow.agents.nodes.retrieve import make_retrieve_node
from insightflow.agents.nodes.rewrite import make_rewrite_node
from insightflow.agents.nodes.route import make_route_node
from insightflow.agents.nodes.verify import (
    NO_CONTEXT_ISSUE,
    check_citations,
    cited_passages,
    make_verify_node,
)
from insightflow.agents.state import AgentState, history_messages, new_turn_input
from insightflow.rag.models import AssembledContext, ContextBudget
from insightflow.rag.retrieval import BudgetedContextAssembler
from tests.fakes import ScriptedChatProvider, StaticRetriever, make_candidate, word_count

HISTORY = [
    ConversationTurn(query="What is InsightFlow?", answer="A RAG platform [1]."),
    ConversationTurn(query="Who builds it?", answer="I do not know."),
]


def context_with(passages: int) -> AssembledContext:
    chunks = [make_candidate(f"chunk-{n}", rank=n) for n in range(1, passages + 1)]
    return AssembledContext(query="q", chunks=chunks, rendered="[1] passage", token_count=2)


def turn(query: str = "What is InsightFlow?", **fields: object) -> AgentState:
    state = new_turn_input(query)
    state.update(fields)  # type: ignore[typeddict-item]
    return state


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
        ("No citation.", 2, ["The answer cites no context passage."]),
        ("Wrong [0] and [3].", 2, ["The answer cites passages that do not exist: [0], [3]."]),
    ],
)
def test_check_citations(answer: str, sources: int, issues: list[str]) -> None:
    assert check_citations(answer, sources) == issues


# --- route -------------------------------------------------------------------------------


async def test_route_without_a_provider_retrieves_the_question_as_written() -> None:
    node = make_route_node(None, history_turns=6)

    assert await node(turn()) == {"route": "retrieve", "retrieval_query": "What is InsightFlow?"}


async def test_route_uses_the_router_decision_and_sends_recent_history() -> None:
    chat = ScriptedChatProvider(
        route='{"route": "retrieve", "standalone_query": "Who builds InsightFlow?"}'
    )
    node = make_route_node(chat, history_turns=1)

    update = await node(turn("Who builds it?", history=HISTORY))

    assert update == {"route": "retrieve", "retrieval_query": "Who builds InsightFlow?"}
    prompt = chat.calls_for("route")[0][1]["content"]
    assert "User: Who builds it?\nAssistant: I do not know." in prompt
    assert "What is InsightFlow?" not in prompt  # only the last turn is sent
    assert prompt.endswith("Latest question: Who builds it?")


async def test_route_can_answer_directly() -> None:
    chat = ScriptedChatProvider(route='{"route": "direct", "standalone_query": "Hello"}')

    update = await make_route_node(chat, history_turns=6)(turn("hello"))

    assert update == {"route": "direct", "retrieval_query": "Hello"}


@pytest.mark.parametrize(
    "reply", ["not json", '{"route": "search", "standalone_query": "x"}', '{"route": "direct"}']
)
async def test_route_falls_back_to_retrieval_for_unusable_replies(reply: str) -> None:
    update = await make_route_node(ScriptedChatProvider(route=reply), history_turns=6)(turn())

    assert update == {"route": "retrieve", "retrieval_query": "What is InsightFlow?"}


async def test_route_keeps_the_question_when_the_standalone_query_is_blank() -> None:
    chat = ScriptedChatProvider(route='{"route": "retrieve", "standalone_query": "   "}')

    update = await make_route_node(chat, history_turns=6)(turn())

    assert update["retrieval_query"] == "What is InsightFlow?"


# --- retrieve ----------------------------------------------------------------------------


async def test_retrieve_uses_the_retrieval_query_and_records_it() -> None:
    retriever = StaticRetriever()
    node = make_retrieve_node(
        retriever,
        BudgetedContextAssembler(word_count),
        top_k=4,
        budget=ContextBudget(max_tokens=100, max_chunks=2),
    )

    update = await node(turn(retrieval_query="standalone", retrieval_queries=["first"]))

    assert [query.text for query in retriever.queries] == ["standalone"]
    assert update["retrieval_queries"] == ["first", "standalone"]
    context = update["context"]
    assert context is not None and context.query == "standalone"


# --- answer ------------------------------------------------------------------------------


async def test_direct_answer_uses_history_and_no_context() -> None:
    chat = ScriptedChatProvider()
    node = make_answer_node(chat, history_turns=1)

    update = await node(turn("thanks", route="direct", history=HISTORY))

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


async def test_verify_skips_when_there_is_no_context() -> None:
    chat = ScriptedChatProvider()

    update = await make_verify_node(chat)(turn(context=context_with(0), answer="x"))

    assert update["verification"] == VerificationResult(
        status="skipped", method="deterministic", issues=[NO_CONTEXT_ISSUE]
    )
    assert chat.calls == []


async def test_verify_fails_bad_citations_without_calling_the_judge() -> None:
    chat = ScriptedChatProvider()

    update = await make_verify_node(chat)(turn(context=context_with(1), answer="Claim [2]."))

    verification = update["verification"]
    assert verification is not None
    assert (verification.status, verification.method) == ("failed", "deterministic")
    assert chat.calls == []


async def test_verify_passes_deterministically_without_a_judge() -> None:
    update = await make_verify_node(None)(turn(context=context_with(1), answer="Claim [1]."))

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
    state = turn(context=context_with(1), answer="Claim [1].")

    update = await make_verify_node(chat)(state)

    assert update["verification"] == expected
    prompt = chat.calls_for("judge")[0][1]["content"]
    assert "Question: What is InsightFlow?" in prompt
    assert prompt.endswith("Answer: Claim [1].")


# --- rewrite -----------------------------------------------------------------------------


async def test_rewrite_produces_a_new_query_and_starts_the_next_attempt() -> None:
    chat = ScriptedChatProvider(rewrite='{"query": "InsightFlow platform purpose"}')
    state = turn(
        retrieval_queries=["What is InsightFlow?"],
        verification=VerificationResult(
            status="failed", method="llm", issues=["unsupported claim"]
        ),
        attempt=1,
    )

    update = await make_rewrite_node(chat)(state)

    assert update == {"retrieval_query": "InsightFlow platform purpose", "attempt": 2}
    prompt = chat.calls_for("rewrite")[0][1]["content"]
    assert "- What is InsightFlow?" in prompt
    assert "- unsupported claim" in prompt


@pytest.mark.parametrize("reply", ["no json", '{"query": "  "}'])
async def test_rewrite_falls_back_to_the_question(reply: str) -> None:
    update = await make_rewrite_node(ScriptedChatProvider(rewrite=reply))(turn(attempt=2))

    assert update == {"retrieval_query": "What is InsightFlow?", "attempt": 3}


# --- finalize ----------------------------------------------------------------------------


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
