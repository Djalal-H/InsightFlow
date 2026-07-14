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

    def __init__(self, model: str) -> None:
        if not model:
            raise ValueError("A LiteLLM chat model identifier is required")
        self._model = model

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        response = await acompletion(model=self._model, messages=list(messages))
        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise ProviderError("The chat provider returned no text content")
        return content

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        response = await acompletion(model=self._model, messages=list(messages), stream=True)
        async for chunk in response:
            content = chunk.choices[0].delta.content
            if isinstance(content, str) and content:
                yield content

