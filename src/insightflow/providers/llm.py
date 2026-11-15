"""Provider-independent chat contract backed by LiteLLM."""

from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Protocol

import litellm
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
from litellm.llms.base_llm.chat.transformation import BaseLLMException

from insightflow.core.exceptions import (
    ProviderConfigurationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from insightflow.core.observability import report_usage

# Never let LiteLLM callbacks or its logging keep prompts and completions (see README "Security").
litellm.turn_off_message_logging = True

ChatMessage = dict[str, Any]


class ChatProvider(Protocol):
    """Contract used by agents that need hosted chat completions."""

    async def complete(self, messages: Sequence[ChatMessage]) -> str: ...

    def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]: ...


class LiteLLMChatProvider:
    """Call any LiteLLM-supported hosted chat provider."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        *,
        max_retries: int = 0,
    ) -> None:
        """`max_retries`: LiteLLM retries of a transient failure (rate limit, 5xx, connection).

        A stream is retried only while it opens, never after its first token.
        """
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
        self._max_retries = max(0, max_retries)

    def _request_options(self) -> dict[str, Any]:
        """Return only configured connection options for LiteLLM."""
        return request_options(self._api_key, self._api_base, self._max_retries)

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        with translate_provider_errors():
            response = await acompletion(
                model=self._model,
                messages=list(messages),
                **self._request_options(),
            )
        report_response_usage(response)

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise ProviderError from exc
        if not isinstance(content, str):
            raise ProviderError
        return content

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        with translate_provider_errors():
            response = await acompletion(
                model=self._model,
                messages=list(messages),
                stream=True,
                **self._request_options(),
            )
            async for chunk in response:
                report_response_usage(chunk)
                content = _delta_content(chunk)
                if content:
                    yield content


def request_options(api_key: str | None, api_base: str | None, max_retries: int) -> dict[str, Any]:
    """The LiteLLM keyword arguments for configured connection and retry options only."""
    options: dict[str, Any] = {}
    if api_key:
        options["api_key"] = api_key
    if api_base:
        options["api_base"] = api_base
    if max_retries > 0:
        options["num_retries"] = max_retries
    return options


@contextmanager
def translate_provider_errors() -> Iterator[None]:
    """Replace LiteLLM exceptions with sanitized application provider errors."""
    try:
        yield
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
    except BaseLLMException as exc:
        # Provider adapters that LiteLLM does not map to its own exceptions (for example its
        # rerank path) raise this base class with the HTTP status.
        raise _error_for_status(exc.status_code) from exc


def _error_for_status(status_code: int) -> ProviderError:
    if status_code in (401, 403, 404):
        return ProviderConfigurationError()
    if status_code == 429:
        return ProviderRateLimitError()
    if status_code in (408, 504):
        return ProviderTimeoutError()
    return ProviderError()


def _token_count(usage: Any, name: str) -> int | None:
    value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def response_usage(response: Any) -> tuple[int | None, int | None]:
    """The prompt and completion token counts of a LiteLLM response or stream chunk."""
    usage = response.get("usage") if isinstance(response, dict) else None
    if usage is None:
        usage = getattr(response, "usage", None)
    if usage is None:
        return None, None
    return _token_count(usage, "prompt_tokens"), _token_count(usage, "completion_tokens")


def report_response_usage(response: Any) -> None:
    """Report the token use of a response to the active provider call, when it has one."""
    prompt_tokens, completion_tokens = response_usage(response)
    if prompt_tokens is not None or completion_tokens is not None:
        report_usage(prompt_tokens, completion_tokens)


def _delta_content(chunk: Any) -> str | None:
    """Return a stream chunk's text; usage-only chunks without choices give None."""
    try:
        content = chunk.choices[0].delta.content
    except (AttributeError, IndexError, KeyError, TypeError):
        return None
    return content if isinstance(content, str) else None
