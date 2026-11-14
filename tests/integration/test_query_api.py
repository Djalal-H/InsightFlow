"""HTTP contract tests for POST /query."""

from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx
import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from insightflow.agents.checkpoint import create_checkpoint_serializer
from insightflow.agents.memory import ConversationMemory, RememberedTurn
from insightflow.agents.nodes.agent import AGENT_SYSTEM_PROMPT
from insightflow.agents.nodes.answer import GROUNDED_SYSTEM_PROMPT, NO_CONTEXT_ANSWER
from insightflow.api.dependencies import (
    get_chat_provider,
    get_checkpointer,
    get_context_assembler,
    get_conversation_memory,
    get_retriever,
)
from insightflow.core.config import Settings, get_settings
from insightflow.core.exceptions import (
    InsightFlowError,
    ProviderConfigurationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    StorageError,
)
from insightflow.main import create_app
from insightflow.providers.llm import ChatMessage, ChatProvider, LiteLLMChatProvider
from insightflow.rag.models import Chunk, RetrievalQuery, RetrievedChunk
from insightflow.rag.retrieval import BudgetedContextAssembler
from tests.fakes import ANSWER, ScriptedChatProvider, search


class FakeChatProvider:
    """Provider-neutral test double that records the route's request."""

    def __init__(self, answer: str = "InsightFlow is a RAG platform [1].") -> None:
        self.answer = answer
        self.messages: list[ChatMessage] | None = None

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        self.messages = list(messages)
        return self.answer

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        del messages
        yield self.answer


class FailingChatProvider(FakeChatProvider):
    """Provider-neutral test double that raises one application error."""

    def __init__(self, error: ProviderError) -> None:
        super().__init__()
        self.error = error

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        del messages
        raise self.error


class FakeRetriever:
    """Returns fixed candidates, or raises one application error."""

    def __init__(
        self,
        candidates: list[RetrievedChunk] | None = None,
        error: InsightFlowError | None = None,
    ) -> None:
        self.candidates = candidates or []
        self.error = error
        self.queries: list[RetrievalQuery] = []

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return self.candidates


def make_candidate() -> RetrievedChunk:
    chunk = Chunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        document_version="v1",
        chunk_index=0,
        content="InsightFlow is an API-first RAG platform.",
        embedding_text="InsightFlow is an API-first RAG platform.",
        page_start=2,
        page_end=2,
        token_count=8,
        content_type="paragraph",
        content_checksum="checksum",
        chunker_name="structure_recursive",
        chunker_version="1.0.0",
    )
    return RetrievedChunk(
        chunk=chunk,
        source_identifier="PDF/overview.pdf",
        source_name="overview.pdf",
        score=0.82,
        rank=1,
    )


# The LLM judge and retries are off in the default test settings. A provider that replies with
# the same text to every prompt makes the agent search the question once and then answer, so a
# request makes three model calls: agent, agent, answer. Tests that need more enable it.
DETERMINISTIC_SETTINGS: dict[str, Any] = {
    "agent_verifier_llm_enabled": False,
    "agent_max_attempts": 1,
}
DEFAULT_NODES = ["agent", "act", "agent", "answer", "verify", "finalize"]


def make_app(
    provider: ChatProvider,
    retriever: FakeRetriever | None = None,
    *,
    checkpointer: BaseCheckpointSaver[str] | None = None,
    **settings: Any,
) -> Any:
    """Build an isolated application with overridden adapters."""
    fake_retriever = retriever or FakeRetriever([make_candidate()])
    values = {**DETERMINISTIC_SETTINGS, **settings}

    application = create_app()
    application.dependency_overrides[get_settings] = lambda: Settings(_env_file=None, **values)
    application.dependency_overrides[get_chat_provider] = lambda: provider
    application.dependency_overrides[get_retriever] = lambda: fake_retriever
    application.dependency_overrides[get_checkpointer] = lambda: checkpointer
    application.dependency_overrides[get_context_assembler] = lambda: BudgetedContextAssembler(
        lambda text: len(text.split())
    )
    return application


def client_for(application: Any) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=application)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def post_query(
    provider: ChatProvider,
    retriever: FakeRetriever | None = None,
) -> httpx.Response:
    """Send a query to an isolated, stateless application."""
    async with client_for(make_app(provider, retriever)) as client:
        return await client.post("/query", json={"query": "What is InsightFlow?"})


def memory_saver() -> InMemorySaver:
    return InMemorySaver(serde=create_checkpoint_serializer())


async def test_query_answers_from_retrieved_context_with_sources() -> None:
    """The route retrieves context, grounds the prompt in it, and returns cited sources."""
    provider = FakeChatProvider()
    retriever = FakeRetriever([make_candidate()])

    response = await post_query(provider, retriever)

    assert response.status_code == 200
    body = response.json()
    assert body["metadata"].pop("duration_ms") >= 0
    [step] = body["metadata"]["steps"]
    assert step.pop("duration_ms") >= 0
    assert len(step.pop("call_id")) == 32
    assert body == {
        "answer": "InsightFlow is a RAG platform [1].",
        "thread_id": None,
        "verification": {"status": "passed", "method": "deterministic", "issues": []},
        "metadata": {
            "attempts": 1,
            "retrieval_queries": ["What is InsightFlow?"],
            "steps": [
                {
                    "attempt": 1,
                    "thought": "Search the documents for the question.",
                    "tool": "search_documents",
                    "server": "insightflow",
                    "arguments": {"query": "What is InsightFlow?"},
                    "status": "ok",
                    "error": None,
                    "truncated": False,
                    "sources": [1],
                }
            ],
            "nodes": DEFAULT_NODES,
            "recalled_turns": 0,
        },
        "sources": [
            {
                "kind": "document",
                "citation": 1,
                "source_identifier": "PDF/overview.pdf",
                "source_name": "overview.pdf",
                "chunk_id": "chunk-1",
                "page_start": 2,
                "page_end": 2,
                "score": 0.82,
            }
        ],
    }
    assert [query.text for query in retriever.queries] == ["What is InsightFlow?"]
    assert retriever.queries[0].top_k == 12
    assert provider.messages is not None
    assert provider.messages[0] == {"role": "system", "content": GROUNDED_SYSTEM_PROMPT}
    user_prompt = provider.messages[1]["content"]
    assert "[1] PDF/overview.pdf (page 2)\nInsightFlow is an API-first RAG platform." in user_prompt
    assert user_prompt.endswith("Question: What is InsightFlow?")


async def test_query_without_retrieved_context_does_not_call_the_answer_model() -> None:
    """An empty search result returns a fixed answer and no sources."""
    provider = FakeChatProvider()

    response = await post_query(provider, FakeRetriever([]))

    assert response.status_code == 200
    body = response.json()
    assert (body["answer"], body["sources"]) == (NO_CONTEXT_ANSWER, [])
    assert body["verification"]["status"] == "skipped"
    assert provider.messages is not None
    assert provider.messages[0] == {"role": "system", "content": AGENT_SYSTEM_PROMPT}


async def test_query_returns_configuration_error_for_missing_model() -> None:
    """Missing model configuration is validated before any hosted request."""
    async def missing_model_provider() -> ChatProvider:
        return LiteLLMChatProvider(model="   ")

    application = create_app()
    application.dependency_overrides[get_settings] = lambda: Settings(_env_file=None)
    application.dependency_overrides[get_chat_provider] = missing_model_provider
    application.dependency_overrides[get_retriever] = lambda: FakeRetriever()
    transport = httpx.ASGITransport(app=application)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/query", json={"query": "What is InsightFlow?"})

    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "provider_configuration_error",
            "message": "The model provider is not configured correctly.",
        },
    }


async def test_query_returns_configuration_error_for_missing_embedding_model() -> None:
    """Retrieval cannot start without an embedding model."""
    application = create_app()
    application.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None, litellm_embedding_model=""
    )
    application.dependency_overrides[get_chat_provider] = lambda: FakeChatProvider()
    transport = httpx.ASGITransport(app=application)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/query", json={"query": "What is InsightFlow?"})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "provider_configuration_error"


async def test_query_maps_storage_failures_to_a_sanitized_response() -> None:
    """A vector-store failure has a stable response without raw details."""
    retriever = FakeRetriever(error=StorageError("secret-storage-detail"))

    response = await post_query(FakeChatProvider(), retriever)

    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "storage_unavailable",
            "message": "The document store is unavailable.",
        },
    }
    assert "secret-storage-detail" not in response.text


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_code", "expected_message"),
    [
        (
            ProviderConfigurationError("secret-provider-detail"),
            503,
            "provider_configuration_error",
            "The model provider is not configured correctly.",
        ),
        (
            ProviderTimeoutError("secret-provider-detail"),
            504,
            "provider_timeout",
            "The model provider timed out.",
        ),
        (
            ProviderRateLimitError("secret-provider-detail"),
            429,
            "provider_rate_limited",
            "The model provider is temporarily rate limited.",
        ),
        (
            ProviderError("secret-provider-detail"),
            502,
            "provider_error",
            "The model provider request failed.",
        ),
    ],
)
async def test_query_maps_provider_failures_to_sanitized_responses(
    error: ProviderError,
    expected_status: int,
    expected_code: str,
    expected_message: str,
) -> None:
    """Every expected provider failure has a stable response without raw details."""
    response = await post_query(FailingChatProvider(error))

    assert response.status_code == expected_status
    assert response.json() == {
        "error": {
            "code": expected_code,
            "message": expected_message,
        },
    }
    assert "secret-provider-detail" not in response.text


def test_query_openapi_documents_stable_errors() -> None:
    """The public API schema includes every stable failure response."""
    responses = create_app().openapi()["paths"]["/query"]["post"]["responses"]

    assert {"429", "502", "503", "504"} <= responses.keys()


def test_query_openapi_documents_stream_and_thread_endpoints() -> None:
    paths = create_app().openapi()["paths"]

    assert "text/event-stream" in paths["/query/stream"]["post"]["responses"]["200"]["content"]
    assert "404" in paths["/threads/{thread_id}"]["get"]["responses"]
    assert "404" in paths["/threads/{thread_id}"]["delete"]["responses"]


async def test_query_retries_and_reports_execution_metadata() -> None:
    """With the judge and retries on, the response describes the whole run."""
    chat = ScriptedChatProvider(
        agent=[search("InsightFlow definition"), ANSWER, search("InsightFlow purpose"), ANSWER],
        answer=["No citation.", "InsightFlow is a RAG platform [1]."],
    )
    retriever = FakeRetriever([make_candidate()])
    application = make_app(
        chat, retriever, agent_verifier_llm_enabled=True, agent_max_attempts=2
    )

    async with client_for(application) as client:
        response = await client.post("/query", json={"query": "What is InsightFlow?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "InsightFlow is a RAG platform [1]."
    assert body["verification"] == {"status": "passed", "method": "llm", "issues": []}
    assert body["metadata"]["attempts"] == 2
    expected_queries = ["InsightFlow definition", "InsightFlow purpose"]
    assert body["metadata"]["retrieval_queries"] == expected_queries
    assert [step["attempt"] for step in body["metadata"]["steps"]] == [1, 2]
    assert [query.text for query in retriever.queries] == expected_queries
    assert chat.roles() == ["agent", "agent", "answer", "agent", "agent", "answer", "judge"]


async def test_query_direct_reply_returns_no_sources() -> None:
    chat = ScriptedChatProvider(agent=ANSWER)
    retriever = FakeRetriever([make_candidate()])

    async with client_for(make_app(chat, retriever)) as client:
        response = await client.post("/query", json={"query": "hello"})

    body = response.json()
    assert body["answer"] == "Hello! Ask me about the indexed documents."
    assert body["sources"] == []
    assert body["metadata"]["steps"] == []
    assert body["metadata"]["nodes"] == ["agent", "answer", "finalize"]
    assert body["verification"] == {"status": "skipped", "method": "none", "issues": []}
    assert retriever.queries == []


async def test_query_creates_a_thread_and_continues_it() -> None:
    """A new thread identifier is returned, and a follow-up on it sees the history."""
    chat = ScriptedChatProvider(answer=["First answer [1].", "Second answer [1]."])
    application = make_app(chat, checkpointer=memory_saver())

    async with client_for(application) as client:
        first = await client.post("/query", json={"query": "What is InsightFlow?"})
        thread_id = first.json()["thread_id"]
        second = await client.post("/query", json={"query": "And why?", "thread_id": thread_id})
        thread = await client.get(f"/threads/{thread_id}")

    assert isinstance(thread_id, str) and thread_id
    assert second.json()["thread_id"] == thread_id
    follow_up_messages = chat.calls_for("answer")[1]
    assert follow_up_messages[1:3] == [
        {"role": "user", "content": "What is InsightFlow?"},
        {"role": "assistant", "content": "First answer [1]."},
    ]
    assert thread.status_code == 200
    assert thread.json() == {
        "thread_id": thread_id,
        "turns": [
            {"query": "What is InsightFlow?", "answer": "First answer [1]."},
            {"query": "And why?", "answer": "Second answer [1]."},
        ],
    }


async def test_query_accepts_a_client_chosen_thread_identifier() -> None:
    application = make_app(ScriptedChatProvider(), checkpointer=memory_saver())

    async with client_for(application) as client:
        response = await client.post("/query", json={"query": "q", "thread_id": "client-1"})

    assert response.json()["thread_id"] == "client-1"


@pytest.mark.parametrize("thread_id", ["", "has space", "x" * 129, "slash/no"])
async def test_query_rejects_invalid_thread_identifiers(thread_id: str) -> None:
    async with client_for(make_app(ScriptedChatProvider())) as client:
        response = await client.post("/query", json={"query": "q", "thread_id": thread_id})

    assert response.status_code == 422


async def test_query_without_a_checkpointer_is_stateless() -> None:
    chat = ScriptedChatProvider()

    async with client_for(make_app(chat)) as client:
        response = await client.post("/query", json={"query": "q", "thread_id": "ignored"})
        thread = await client.get("/threads/ignored")

    assert response.json()["thread_id"] is None
    assert thread.status_code == 404


async def test_thread_endpoints_report_unknown_threads() -> None:
    async with client_for(make_app(ScriptedChatProvider(), checkpointer=memory_saver())) as client:
        read = await client.get("/threads/missing")
        delete = await client.delete("/threads/missing")

    for response in (read, delete):
        assert response.status_code == 404
        assert response.json() == {
            "error": {
                "code": "thread_not_found",
                "message": "The conversation thread does not exist.",
            }
        }


async def test_thread_can_be_deleted() -> None:
    application = make_app(ScriptedChatProvider(), checkpointer=memory_saver())

    async with client_for(application) as client:
        await client.post("/query", json={"query": "q", "thread_id": "t1"})
        deleted = await client.delete("/threads/t1")
        read = await client.get("/threads/t1")

    assert deleted.status_code == 204
    assert read.status_code == 404


# --- Stage 6: request filters and conversation memory ----------------------------------------


async def test_query_filters_limit_document_search() -> None:
    retriever = FakeRetriever([make_candidate()])
    application = make_app(FakeChatProvider(), retriever)

    async with client_for(application) as client:
        response = await client.post(
            "/query",
            json={
                "query": "What is InsightFlow?",
                "filters": {"file_types": ["pdf", "pdf"], "sources": ["PDF/overview.pdf"]},
            },
        )

    assert response.status_code == 200
    [query] = retriever.queries
    assert query.source_types == ["application/pdf", "application/x-pdf"]
    assert query.source_identifiers == ["PDF/overview.pdf"]


@pytest.mark.parametrize(
    "filters",
    [{"file_types": ["exe"]}, {"file_types": []}, {"sources": []}, {"sources": [""]}],
)
async def test_query_rejects_invalid_filters(filters: dict[str, Any]) -> None:
    async with client_for(make_app(FakeChatProvider())) as client:
        response = await client.post("/query", json={"query": "q", "filters": filters})

    assert response.status_code == 422


class RecordingMemoryStore:
    """An in-process `MemoryStore` that returns every older turn of the thread."""

    def __init__(self) -> None:
        self.turns: dict[tuple[str, int], tuple[str, str]] = {}

    async def save_turn(
        self, thread_id: str, turn_index: int, vector: Sequence[float], *, query: str, answer: str
    ) -> None:
        self.turns[(thread_id, turn_index)] = (query, answer)

    async def search_turns(
        self,
        thread_id: str,
        vector: Sequence[float],
        *,
        limit: int,
        before_turn: int,
        score_threshold: float | None = None,
    ) -> list[RememberedTurn]:
        return [
            RememberedTurn(turn_index=index, query=query, answer=answer, score=1.0)
            for (thread, index), (query, answer) in self.turns.items()
            if thread == thread_id and index < before_turn
        ][:limit]

    async def delete_thread(self, thread_id: str) -> None:
        self.turns = {key: turn for key, turn in self.turns.items() if key[0] != thread_id}


class ConstantEmbeddingProvider:
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [1.0]

    def count_tokens(self, text: str) -> int:
        return len(text.split())


async def test_memory_recalls_turns_outside_the_history_window_and_is_deleted_with_the_thread(
) -> None:
    store = RecordingMemoryStore()
    memory = ConversationMemory(ConstantEmbeddingProvider(), store, top_k=1)
    chat = ScriptedChatProvider(answer=["First [1].", "Second [1].", "Third [1]."])
    application = make_app(chat, checkpointer=memory_saver(), agent_history_turns=1)
    application.dependency_overrides[get_conversation_memory] = lambda: memory

    async with client_for(application) as client:
        for question in ("One?", "Two?"):
            await client.post("/query", json={"query": question, "thread_id": "t1"})
        third = await client.post("/query", json={"query": "Three?", "thread_id": "t1"})
        stored_before_delete = dict(store.turns)
        deleted = await client.delete("/threads/t1")

    metadata = third.json()["metadata"]
    assert metadata["recalled_turns"] == 1
    assert metadata["nodes"] == ["recall", *DEFAULT_NODES]
    assert chat.calls_for("answer")[2][1:5] == [
        {"role": "user", "content": "One?"},
        {"role": "assistant", "content": "First [1]."},
        {"role": "user", "content": "Two?"},
        {"role": "assistant", "content": "Second [1]."},
    ]
    assert sorted(stored_before_delete) == [("t1", 0), ("t1", 1), ("t1", 2)]
    assert deleted.status_code == 204
    assert store.turns == {}


async def test_memory_is_not_used_without_a_checkpointer() -> None:
    store = RecordingMemoryStore()
    memory = ConversationMemory(ConstantEmbeddingProvider(), store)
    application = make_app(ScriptedChatProvider())
    application.dependency_overrides[get_conversation_memory] = lambda: memory

    async with client_for(application) as client:
        response = await client.post("/query", json={"query": "One?"})

    assert response.json()["metadata"]["nodes"] == DEFAULT_NODES
    assert store.turns == {}


async def test_memory_dependency_is_none_by_default() -> None:
    disabled = get_conversation_memory(Settings(_env_file=None, memory_enabled=False))

    assert await anext(disabled) is None
