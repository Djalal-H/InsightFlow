"""Provider wrappers that record each call in the active run (see `core.observability`).

The wrappers time every call, record its outcome and error code, check the run's budget before
the call, and let the wrapped adapter report token use (`report_usage`). They work with any
implementation of the protocols, so test doubles are recorded the same way as LiteLLM.
"""

import time
from collections.abc import AsyncIterator, Sequence

from insightflow.core.observability import (
    finish_provider_call,
    start_provider_call,
    usage_target,
)
from insightflow.providers.embeddings import EmbeddingProvider
from insightflow.providers.llm import ChatMessage, ChatProvider
from insightflow.providers.rerank import RerankProvider, RerankResult


class InstrumentedChatProvider:
    def __init__(self, inner: ChatProvider, *, model: str) -> None:
        self.inner = inner
        self.model = model

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        call = start_provider_call("chat", "complete", self.model, list(messages))
        started = time.perf_counter()
        try:
            with usage_target(call):
                output = await self.inner.complete(messages)
        except BaseException as exc:
            finish_provider_call(call, started, error=exc)
            raise
        call.output = output
        finish_provider_call(call, started)
        return output

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        call = start_provider_call("chat", "stream", self.model, list(messages))
        started = time.perf_counter()
        pieces: list[str] = []
        iterator = self.inner.stream(messages).__aiter__()
        error: BaseException | None = None
        try:
            while True:
                # The usage target is set around each step only, so it never leaks to the caller.
                with usage_target(call):
                    try:
                        piece = await anext(iterator)
                    except StopAsyncIteration:
                        break
                pieces.append(piece)
                yield piece
        except BaseException as exc:
            error = exc
            raise
        finally:
            aclose = getattr(iterator, "aclose", None)
            if aclose is not None:
                await aclose()
            call.output = "".join(pieces)
            finish_provider_call(call, started, error=error)


class InstrumentedEmbeddingProvider:
    def __init__(self, inner: EmbeddingProvider, *, model: str) -> None:
        self.inner = inner
        self.model = model

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        call = start_provider_call("embedding", "embed_documents", self.model)
        started = time.perf_counter()
        try:
            with usage_target(call):
                vectors = await self.inner.embed_documents(texts)
        except BaseException as exc:
            finish_provider_call(call, started, error=exc)
            raise
        finish_provider_call(call, started)
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        call = start_provider_call("embedding", "embed_query", self.model)
        started = time.perf_counter()
        try:
            with usage_target(call):
                vector = await self.inner.embed_query(text)
        except BaseException as exc:
            finish_provider_call(call, started, error=exc)
            raise
        finish_provider_call(call, started)
        return vector

    def count_tokens(self, text: str) -> int:
        return self.inner.count_tokens(text)


class InstrumentedRerankProvider:
    def __init__(self, inner: RerankProvider, *, model: str) -> None:
        self.inner = inner
        self.model = model

    async def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int
    ) -> list[RerankResult]:
        call = start_provider_call("rerank", "rerank", self.model)
        started = time.perf_counter()
        try:
            with usage_target(call):
                results = await self.inner.rerank(query, documents, top_n=top_n)
        except BaseException as exc:
            finish_provider_call(call, started, error=exc)
            raise
        finish_provider_call(call, started)
        return results


def instrument_chat(provider: ChatProvider, *, model: str) -> ChatProvider:
    """Wrap `provider` once; an instrumented provider is returned unchanged."""
    if isinstance(provider, InstrumentedChatProvider):
        return provider
    return InstrumentedChatProvider(provider, model=model)


def instrument_embedding(provider: EmbeddingProvider, *, model: str) -> EmbeddingProvider:
    if isinstance(provider, InstrumentedEmbeddingProvider):
        return provider
    return InstrumentedEmbeddingProvider(provider, model=model)


def instrument_rerank(provider: RerankProvider, *, model: str) -> RerankProvider:
    if isinstance(provider, InstrumentedRerankProvider):
        return provider
    return InstrumentedRerankProvider(provider, model=model)
