"""Provider-independent chat contract backed by LiteLLM."""

from collections.abc import AsyncIterator, Sequence
from typing import Any, Protocol

from litellm import acompletion, get_llm_provider
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    APIResponseValidationError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
    UnprocessableEntityError,
)

from insightflow.core.exceptions import (
    ProviderConfigurationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)

ChatMessage = dict[str, Any]


class ChatProvider(Protocol):
    """Contract used by agents that need hosted chat completions."""

    async def complete(self, messages: Sequence[ChatMessage]) -> str: ...

    def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]: ...


class LiteLLMChatProvider:
    """Call any LiteLLM-supported hosted chat provider."""

    def __init__(self, model: str, api_key: str | None = None, api_base: str | None = None) -> None:
        normalized_model = model.strip()
        if not normalized_model:
            raise ProviderConfigurationError

        try:
            resolved_model, provider, _, _ = get_llm_provider(
                model=normalized_model,
                api_key=api_key,
                api_base=api_base,
            )
        except BadRequestError as exc:
            raise ProviderConfigurationError from exc

        if not resolved_model.strip() or not provider:
            raise ProviderConfigurationError

        self._model = normalized_model
        self._api_key = api_key
        self._api_base = api_base

    def _request_options(self) -> dict[str, str]:
        """Return only configured connection options for LiteLLM."""
        options: dict[str, str] = {}
        if self._api_key:
            options["api_key"] = self._api_key
        if self._api_base:
            options["api_base"] = self._api_base
        return options

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        try:
            response = await acompletion(
                model=self._model,
                messages=list(messages),
                **self._request_options(),
            )
        except Timeout as exc:
            raise ProviderTimeoutError from exc
        except RateLimitError as exc:
            raise ProviderRateLimitError from exc
        except (
            AuthenticationError,
            BadRequestError,
            NotFoundError,
            PermissionDeniedError,
        ) as exc:
            raise ProviderConfigurationError from exc
        except (
            APIConnectionError,
            APIError,
            APIResponseValidationError,
            InternalServerError,
            ServiceUnavailableError,
            UnprocessableEntityError,
        ) as exc:
            raise ProviderError from exc

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise ProviderError from exc
        if not isinstance(content, str):
            raise ProviderError
        return content

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        response = await acompletion(
            model=self._model,
            messages=list(messages),
            stream=True,
            **self._request_options(),
        )
        async for chunk in response:
            content = chunk.choices[0].delta.content
            if isinstance(content, str) and content:
                yield content
