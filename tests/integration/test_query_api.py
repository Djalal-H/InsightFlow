"""HTTP contract tests for POST /query."""

from collections.abc import AsyncIterator, Sequence

import httpx
import pytest

from insightflow.agents.nodes.answer import GROUNDED_SYSTEM_PROMPT, NO_CONTEXT_ANSWER
from insightflow.api.dependencies import (
    get_chat_provider,
    get_context_assembler,
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


async def post_query(
    provider: ChatProvider,
    retriever: FakeRetriever | None = None,
) -> httpx.Response:
    """Send a query to an isolated application with overridden adapters."""
    fake_retriever = retriever or FakeRetriever([make_candidate()])

    application = create_app()
    application.dependency_overrides[get_settings] = lambda: Settings(_env_file=None)
    application.dependency_overrides[get_chat_provider] = lambda: provider
    application.dependency_overrides[get_retriever] = lambda: fake_retriever
    application.dependency_overrides[get_context_assembler] = lambda: BudgetedContextAssembler(
        lambda text: len(text.split())
    )
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/query", json={"query": "What is InsightFlow?"})


async def test_query_answers_from_retrieved_context_with_sources() -> None:
    """The route retrieves context, grounds the prompt in it, and returns cited sources."""
    provider = FakeChatProvider()
    retriever = FakeRetriever([make_candidate()])

    response = await post_query(provider, retriever)

    assert response.status_code == 200
    assert response.json() == {
        "answer": "InsightFlow is a RAG platform [1].",
        "sources": [
            {
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


async def test_query_without_retrieved_context_does_not_call_the_model() -> None:
    """An empty retrieval result returns a fixed answer and no sources."""
    provider = FakeChatProvider()

    response = await post_query(provider, FakeRetriever([]))

    assert response.status_code == 200
    assert response.json() == {"answer": NO_CONTEXT_ANSWER, "sources": []}
    assert provider.messages is None


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
