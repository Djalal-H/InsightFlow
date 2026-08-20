"""Provider-independent embedding contract backed by LiteLLM."""

from collections.abc import Sequence
from typing import Protocol

from litellm import aembedding, token_counter

from insightflow.core.exceptions import ProviderError


class EmbeddingProvider(Protocol):
    """Contract used by ingestion and retrieval to create hosted embeddings."""

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...

    def count_tokens(self, text: str) -> int: ...


class LiteLLMEmbeddingProvider:
    """Generate embeddings with a LiteLLM-supported hosted provider."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        normalized_model = model.strip()
        if not normalized_model:
            raise ValueError("A LiteLLM embedding model identifier is required")
        self._model = normalized_model
        self._api_key = api_key
        self._api_base = api_base

    def _request_options(self) -> dict[str, str]:
        """Return only explicitly configured connection options for LiteLLM."""
        options: dict[str, str] = {}
        if self._api_key:
            options["api_key"] = self._api_key
        if self._api_base:
            options["api_base"] = self._api_base
        return options

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed document inputs while preserving their input order."""
        response = await aembedding(
            model=self._model,
            input=list(texts),
            **self._request_options(),
        )
        vectors = [item["embedding"] for item in response.data]
        if len(vectors) != len(texts):
            raise ProviderError("The embedding provider returned an unexpected vector count")
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        """Embed one query through a query-specific domain method."""
        vectors = await self.embed_documents([text])
        return vectors[0]

    def count_tokens(self, text: str) -> int:
        """Count tokens using the tokenizer associated with the configured model."""
        try:
            count = token_counter(model=self._model, text=text)
        except Exception as exc:
            raise ProviderError("Token counting failed for the configured embedding model") from exc
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ProviderError("Token counting returned an invalid result")
        return count
