"""Provider-independent embedding contract backed by LiteLLM."""

from collections.abc import Sequence
from typing import Protocol

from litellm import aembedding

from insightflow.core.exceptions import ProviderError


class EmbeddingProvider(Protocol):
    """Contract used by ingestion and retrieval to create hosted embeddings."""

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class LiteLLMEmbeddingProvider:
    """Generate embeddings with a LiteLLM-supported hosted provider."""

    def __init__(self, model: str) -> None:
        if not model:
            raise ValueError("A LiteLLM embedding model identifier is required")
        self._model = model

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed document inputs while preserving their input order."""
        response = await aembedding(model=self._model, input=list(texts))
        vectors = [item["embedding"] for item in response.data]
        if len(vectors) != len(texts):
            raise ProviderError("The embedding provider returned an unexpected vector count")
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        """Embed one query through a query-specific domain method."""
        vectors = await self.embed_documents([text])
        return vectors[0]
