"""Provider-independent chat contract backed by LiteLLM."""

from collections.abc import AsyncIterator, Sequence
from typing import Any, Protocol

from litellm import acompletion

from insightflow.core.exceptions import ProviderError

ChatMessage = dict[str, Any]


class ChatProvider(Protocol):
    """Contract used by agents that need hosted chat completions."""

    async def complete(self, messages: Sequence[ChatMessage]) -> str: ...

    def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]: ...


class LiteLLMChatProvider:
    """Call any LiteLLM-supported hosted chat provider."""

    def __init__(self, model: str, api_key: str | None = None, api_base: str | None = None) -> None:
        if not model:
            raise ValueError("A LiteLLM chat model identifier is required")
        self._model = model
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
        response = await acompletion(
            model=self._model,
            messages=list(messages),
            **self._request_options(),
        )
        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise ProviderError("The chat provider returned no text content")
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
