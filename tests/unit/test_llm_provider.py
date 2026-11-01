"""Unit tests for the LiteLLM chat adapter boundary."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from litellm import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
    Timeout,
)

from insightflow.core.exceptions import (
    ProviderConfigurationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from insightflow.providers.llm import LiteLLMChatProvider

MODEL = "openai/test-model"
MESSAGES = [{"role": "user", "content": "Hello"}]


def build_provider() -> LiteLLMChatProvider:
    """Build an adapter while isolating model resolution from LiteLLM registries."""
    with patch(
        "insightflow.providers.llm.get_llm_provider",
        return_value=("test-model", "openai", None, None),
    ):
        return LiteLLMChatProvider(
            model=MODEL,
            api_key="test-api-key",
            api_base="https://provider.test/v1",
        )


@pytest.mark.parametrize("model", ["", "   "])
def test_provider_rejects_missing_model(model: str) -> None:
    """Empty and whitespace-only model settings fail before a provider call."""
    with pytest.raises(ProviderConfigurationError):
        LiteLLMChatProvider(model=model)


def test_provider_rejects_unresolvable_model() -> None:
    """LiteLLM provider-resolution failures become application configuration errors."""
    provider_error = BadRequestError(
        message="secret invalid model detail",
        model="unknown/model",
        llm_provider="unknown",
    )
    with (
        patch(
            "insightflow.providers.llm.get_llm_provider",
            side_effect=provider_error,
        ),
        pytest.raises(ProviderConfigurationError) as raised,
    ):
        LiteLLMChatProvider(model="unknown/model")

    assert raised.value.__cause__ is provider_error
    assert not raised.value.args


def test_provider_rejects_provider_prefix_without_model() -> None:
    """A recognized provider prefix still requires a non-empty model name."""
    with (
        patch(
            "insightflow.providers.llm.get_llm_provider",
            return_value=("", "openai", None, None),
        ),
        pytest.raises(ProviderConfigurationError),
    ):
        LiteLLMChatProvider(model="openai/")


@pytest.mark.asyncio
async def test_provider_returns_text_and_passes_connection_options() -> None:
    """A successful completion forwards neutral messages and configured options."""
    provider = build_provider()
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Provider answer"))],
    )

    with patch(
        "insightflow.providers.llm.acompletion",
        new_callable=AsyncMock,
        return_value=response,
    ) as completion:
        result = await provider.complete(MESSAGES)

    assert result == "Provider answer"
    completion.assert_awaited_once_with(
        model=MODEL,
        messages=MESSAGES,
        api_key="test-api-key",
        api_base="https://provider.test/v1",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("litellm_error", "application_error"),
    [
        (
            AuthenticationError(
                message="secret authentication detail",
                llm_provider="openai",
                model="test-model",
            ),
            ProviderConfigurationError,
        ),
        (
            BadRequestError(
                message="secret invalid model detail",
                model="test-model",
                llm_provider="openai",
            ),
            ProviderConfigurationError,
        ),
        (
            Timeout(
                message="secret timeout detail",
                model="test-model",
                llm_provider="openai",
            ),
            ProviderTimeoutError,
        ),
        (
            RateLimitError(
                message="secret rate-limit detail",
                llm_provider="openai",
                model="test-model",
            ),
            ProviderRateLimitError,
        ),
        (
            APIConnectionError(
                message="secret connection detail",
                llm_provider="openai",
                model="test-model",
            ),
            ProviderError,
        ),
    ],
)
async def test_provider_translates_litellm_failures(
    litellm_error: Exception,
    application_error: type[ProviderError],
) -> None:
    """LiteLLM-specific failures do not cross the provider adapter boundary."""
    provider = build_provider()

    with (
        patch(
            "insightflow.providers.llm.acompletion",
            new_callable=AsyncMock,
            side_effect=litellm_error,
        ),
        pytest.raises(application_error) as raised,
    ):
        await provider.complete(MESSAGES)

    assert raised.value.__cause__ is litellm_error
    assert not raised.value.args


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(choices=[]),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))]),
    ],
)
async def test_provider_rejects_malformed_responses(response: object) -> None:
    """Missing text content is exposed only as a generic provider failure."""
    provider = build_provider()

    with (
        patch(
            "insightflow.providers.llm.acompletion",
            new_callable=AsyncMock,
            return_value=response,
        ),
        pytest.raises(ProviderError) as raised,
    ):
        await provider.complete(MESSAGES)

    assert not raised.value.args


class FakeStream:
    """Async iterator over stream chunks that can fail part of the way through."""

    def __init__(self, chunks: list[object], error: Exception | None = None) -> None:
        self.chunks = chunks
        self.error = error

    def __aiter__(self) -> "FakeStream":
        return self

    async def __anext__(self) -> object:
        if self.chunks:
            return self.chunks.pop(0)
        if self.error is not None:
            raise self.error
        raise StopAsyncIteration


def delta(content: object) -> object:
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])


async def test_provider_streams_text_and_skips_empty_chunks() -> None:
    provider = build_provider()
    stream = FakeStream(
        [delta("Hello"), delta(None), SimpleNamespace(choices=[]), delta(""), delta(" world")]
    )

    with patch(
        "insightflow.providers.llm.acompletion", new_callable=AsyncMock, return_value=stream
    ) as completion:
        pieces = [piece async for piece in provider.stream(MESSAGES)]

    assert pieces == ["Hello", " world"]
    completion.assert_awaited_once_with(
        model=MODEL,
        messages=MESSAGES,
        stream=True,
        api_key="test-api-key",
        api_base="https://provider.test/v1",
    )


async def test_provider_translates_failures_when_a_stream_starts() -> None:
    provider = build_provider()
    error = RateLimitError(message="secret", llm_provider="openai", model="test-model")

    with (
        patch("insightflow.providers.llm.acompletion", new_callable=AsyncMock, side_effect=error),
        pytest.raises(ProviderRateLimitError) as raised,
    ):
        [piece async for piece in provider.stream(MESSAGES)]

    assert raised.value.__cause__ is error


async def test_provider_translates_failures_during_a_stream() -> None:
    provider = build_provider()
    error = APIConnectionError(message="secret", llm_provider="openai", model="test-model")
    pieces: list[str] = []

    with (
        patch(
            "insightflow.providers.llm.acompletion",
            new_callable=AsyncMock,
            return_value=FakeStream([delta("partial")], error),
        ),
        pytest.raises(ProviderError) as raised,
    ):
        async for piece in provider.stream(MESSAGES):
            pieces.append(piece)

    assert pieces == ["partial"]
    assert raised.value.__cause__ is error
    assert not raised.value.args
