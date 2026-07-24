"""HTTP contract tests for POST /query."""

from collections.abc import AsyncIterator, Sequence

import httpx
import pytest

from insightflow.api.dependencies import get_chat_provider
from insightflow.core.exceptions import (
    ProviderConfigurationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from insightflow.main import create_app
from insightflow.providers.llm import ChatMessage, ChatProvider, LiteLLMChatProvider


class FakeChatProvider:
    """Provider-neutral test double that records the route's request."""

    def __init__(self, answer: str = "This is a mocked answer.") -> None:
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


async def post_query(provider: FakeChatProvider) -> httpx.Response:
    """Send a query to an isolated application with an overridden provider."""
    async def override_chat_provider() -> FakeChatProvider:
        return provider

    application = create_app()
    application.dependency_overrides[get_chat_provider] = override_chat_provider
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/query", json={"query": "What is InsightFlow?"})


@pytest.mark.asyncio
async def test_query_uses_provider_contract_and_returns_answer() -> None:
    """The route works with a protocol-compatible provider and no LiteLLM patch."""
    provider = FakeChatProvider()

    response = await post_query(provider)

    assert response.status_code == 200
    assert response.json() == {
        "answer": "This is a mocked answer.",
        "sources": [],
    }
    assert provider.messages == [
        {"role": "user", "content": "What is InsightFlow?"},
    ]


@pytest.mark.asyncio
async def test_query_returns_configuration_error_for_missing_model() -> None:
    """Missing model configuration is validated before any hosted request."""
    async def missing_model_provider() -> ChatProvider:
        return LiteLLMChatProvider(model="   ")

    application = create_app()
    application.dependency_overrides[get_chat_provider] = missing_model_provider
    transport = httpx.ASGITransport(app=application)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/query", json={"query": "What is InsightFlow?"})

    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "provider_configuration_error",
            "message": "The chat provider is not configured correctly.",
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_status", "expected_code", "expected_message"),
    [
        (
            ProviderConfigurationError("secret-provider-detail"),
            503,
            "provider_configuration_error",
            "The chat provider is not configured correctly.",
        ),
        (
            ProviderTimeoutError("secret-provider-detail"),
            504,
            "provider_timeout",
            "The chat provider timed out.",
        ),
        (
            ProviderRateLimitError("secret-provider-detail"),
            429,
            "provider_rate_limited",
            "The chat provider is temporarily rate limited.",
        ),
        (
            ProviderError("secret-provider-detail"),
            502,
            "provider_error",
            "The chat provider request failed.",
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


def test_query_openapi_documents_stable_provider_errors() -> None:
    """The public API schema includes every stable provider failure response."""
    responses = create_app().openapi()["paths"]["/query"]["post"]["responses"]

    assert {"429", "502", "503", "504"} <= responses.keys()
