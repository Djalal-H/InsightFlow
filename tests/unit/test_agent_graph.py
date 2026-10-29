"""Tests for the retrieve → answer LangGraph workflow."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from insightflow.agents.graph import build_graph
from insightflow.agents.nodes.answer import NO_CONTEXT_ANSWER, make_answer_node
from insightflow.agents.nodes.retrieve import make_retrieve_node
from insightflow.providers.llm import ChatMessage
from insightflow.rag.models import Chunk, ContextBudget, RetrievalQuery, RetrievedChunk
from insightflow.rag.retrieval import BudgetedContextAssembler


class RecordingChatProvider:
    def __init__(self) -> None:
        self.calls: list[list[ChatMessage]] = []

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        self.calls.append(list(messages))
        return "Grounded answer [1]."

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        del messages
        yield ""


class StaticRetriever:
    def __init__(self, candidates: list[RetrievedChunk]) -> None:
        self.candidates = candidates
        self.queries: list[RetrievalQuery] = []

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        self.queries.append(query)
        return self.candidates


def candidate() -> RetrievedChunk:
    chunk = Chunk(
        chunk_id="chunk-1",
        document_id="doc",
        document_version="v1",
        chunk_index=0,
        content="Stored passage.",
        embedding_text="Stored passage.",
        token_count=2,
        content_type="paragraph",
        content_checksum="checksum",
        chunker_name="structure_recursive",
        chunker_version="1.0.0",
    )
    return RetrievedChunk(
        chunk=chunk, source_identifier="a.txt", source_name="a.txt", score=0.5, rank=1
    )


def make_graph(retriever: StaticRetriever, chat: RecordingChatProvider):  # type: ignore[no-untyped-def]
    retrieve = make_retrieve_node(
        retriever,
        BudgetedContextAssembler(lambda text: len(text.split())),
        top_k=4,
        budget=ContextBudget(max_tokens=100, max_chunks=2),
    )
    return build_graph(retrieve, make_answer_node(chat))


async def test_graph_retrieves_then_answers_from_context() -> None:
    retriever = StaticRetriever([candidate()])
    chat = RecordingChatProvider()

    result = await make_graph(retriever, chat).ainvoke({"query": "What is stored?"})

    assert result["answer"] == "Grounded answer [1]."
    assert [item.chunk.chunk_id for item in result["context"].chunks] == ["chunk-1"]
    assert retriever.queries == [RetrievalQuery(text="What is stored?", top_k=4)]
    assert "[1] a.txt\nStored passage." in chat.calls[0][1]["content"]


async def test_graph_answers_without_the_model_when_nothing_is_retrieved() -> None:
    chat = RecordingChatProvider()

    result = await make_graph(StaticRetriever([]), chat).ainvoke({"query": "Anything?"})

    assert result["answer"] == NO_CONTEXT_ANSWER
    assert result["context"].chunks == []
    assert chat.calls == []
