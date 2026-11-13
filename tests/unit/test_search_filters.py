"""Tests for metadata filters in document search and for the agent prompt options."""

from __future__ import annotations

from typing import Any

import pytest

from insightflow.agents.document_search import (
    MIME_TYPES_BY_FILE_TYPE,
    SEARCH_SPEC,
    DocumentSearch,
    search_spec,
)
from insightflow.agents.models import AgentAction
from insightflow.agents.nodes.act import make_act_node
from insightflow.agents.nodes.agent import MULTI_HOP_HINT, make_agent_node
from insightflow.agents.state import new_turn_input
from insightflow.rag.models import ContextBudget
from insightflow.rag.retrieval import BudgetedContextAssembler
from insightflow.tools.registry import ToolSet
from tests.fakes import ScriptedChatProvider, StaticRetriever, action, word_count


def document_search(retriever: StaticRetriever, *, filters: bool = True) -> DocumentSearch:
    return DocumentSearch(
        retriever,
        BudgetedContextAssembler(word_count),
        top_k=4,
        budget=ContextBudget(max_tokens=100, max_chunks=2),
        filters_enabled=filters,
    )


def test_filter_arguments_are_in_the_tool_schema_only_when_enabled() -> None:
    plain = SEARCH_SPEC.input_schema["properties"]
    filtered = search_spec(filters=True)

    assert set(plain) == {"query"}
    assert set(filtered.input_schema["properties"]) == {"query", "file_types", "sources"}
    assert filtered.input_schema["required"] == ["query"]
    assert "file_types or sources" in filtered.description
    assert "file_types" not in SEARCH_SPEC.description


async def test_tool_filters_become_retrieval_filters() -> None:
    retriever = StaticRetriever()
    arguments = {"query": "fees", "file_types": ["pdf", "md"], "sources": ["PDF/handbook.pdf"]}

    result, passages = await document_search(retriever).search(arguments)

    assert result.status == "ok" and passages
    [query] = retriever.queries
    assert query.source_types == [
        "application/pdf",
        "application/x-pdf",
        *MIME_TYPES_BY_FILE_TYPE["md"],
    ]
    assert query.source_identifiers == ["PDF/handbook.pdf"]


async def test_search_without_filters_sends_none() -> None:
    retriever = StaticRetriever()

    await document_search(retriever).search({"query": "fees"})

    assert retriever.queries[0].source_types is None
    assert retriever.queries[0].source_identifiers is None


@pytest.mark.parametrize(
    "arguments",
    [
        {"query": "fees", "file_types": ["exe"]},
        {"query": "fees", "file_types": []},
        {"query": "fees", "sources": "PDF/handbook.pdf"},
    ],
)
async def test_invalid_filter_arguments_are_tool_errors(arguments: dict[str, Any]) -> None:
    retriever = StaticRetriever()

    result, passages = await document_search(retriever).search(arguments)

    assert (result.status, result.error, passages) == ("error", "invalid_arguments", [])
    assert retriever.queries == []


async def test_filter_arguments_are_refused_when_filters_are_disabled() -> None:
    retriever = StaticRetriever()

    result, _ = await document_search(retriever, filters=False).search(
        {"query": "fees", "file_types": ["pdf"]}
    )

    assert result.error == "invalid_arguments"


async def test_request_scope_limits_every_search_and_tool_filters_only_narrow_it() -> None:
    retriever = StaticRetriever()
    search = document_search(retriever)
    scope = {"file_types": ["pdf", "docx"], "sources": ["PDF/a.pdf", "PDF/b.pdf"]}

    await search.search({"query": "fees"}, scope=scope)
    await search.search(
        {"query": "fees", "file_types": ["pdf", "md"], "sources": ["PDF/b.pdf"]}, scope=scope
    )
    outside, passages = await search.search(
        {"query": "fees", "file_types": ["txt"]}, scope=scope
    )

    first, second = retriever.queries
    assert first.source_types == [
        *MIME_TYPES_BY_FILE_TYPE["pdf"],
        *MIME_TYPES_BY_FILE_TYPE["docx"],
    ]
    assert first.source_identifiers == ["PDF/a.pdf", "PDF/b.pdf"]
    assert second.source_types == MIME_TYPES_BY_FILE_TYPE["pdf"]
    assert second.source_identifiers == ["PDF/b.pdf"]
    assert (outside.status, outside.output, passages) == ("ok", "0 passages", [])
    assert len(retriever.queries) == 2


async def test_act_node_passes_the_request_scope_to_the_search() -> None:
    retriever = StaticRetriever()
    node = make_act_node(document_search(retriever, filters=False), ToolSet())
    state = new_turn_input("q", search_scope={"sources": ["TXT/dracula.txt"]})
    state["pending_action"] = AgentAction(
        action="search_documents", arguments={"query": "castle"}
    )

    await node(state)

    assert retriever.queries[0].source_identifiers == ["TXT/dracula.txt"]


async def test_agent_prompt_shows_filters_scope_and_the_multi_hop_hint() -> None:
    chat = ScriptedChatProvider(agent=action("search_documents", query="x"))
    node = make_agent_node(
        chat,
        ToolSet(),
        max_steps=2,
        history_turns=0,
        search_spec=search_spec(filters=True),
        multi_hop=True,
    )

    await node(new_turn_input("q", search_scope={"file_types": ["pdf"]}))

    prompt = chat.calls_for("agent")[0][1]["content"]
    assert '"file_types"' in prompt
    assert "The user limited document search to file_types: pdf." in prompt
    assert MULTI_HOP_HINT in prompt


async def test_agent_prompt_is_unchanged_by_default() -> None:
    chat = ScriptedChatProvider(agent=action("search_documents", query="x"))
    node = make_agent_node(chat, ToolSet(), max_steps=2, history_turns=0)

    await node(new_turn_input("q"))

    prompt = chat.calls_for("agent")[0][1]["content"]
    assert MULTI_HOP_HINT not in prompt
    assert "limited document search" not in prompt
    assert '"file_types"' not in prompt
