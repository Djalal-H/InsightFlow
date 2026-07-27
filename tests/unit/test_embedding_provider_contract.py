"""Tests for the query/document embedding contract."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from insightflow.providers.embeddings import (
    EmbeddingProvider,
    LiteLLMEmbeddingProvider,
)


@pytest.mark.asyncio
async def test_embedding_provider_separates_document_and_query_methods() -> None:
    """The adapter exposes distinct methods even when one provider endpoint backs both."""
    provider = LiteLLMEmbeddingProvider("hosted/test-embedding")
    response = SimpleNamespace(
        data=[
            {"embedding": [0.1, 0.2]},
            {"embedding": [0.3, 0.4]},
        ]
    )

    with patch(
        "insightflow.providers.embeddings.aembedding",
        new_callable=AsyncMock,
        return_value=response,
    ) as embedding:
        vectors = await provider.embed_documents(["document one", "document two"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    embedding.assert_awaited_once_with(
        model="hosted/test-embedding",
        input=["document one", "document two"],
    )

    query_response = SimpleNamespace(data=[{"embedding": [0.5, 0.6]}])
    with patch(
        "insightflow.providers.embeddings.aembedding",
        new_callable=AsyncMock,
        return_value=query_response,
    ) as embedding:
        vector = await provider.embed_query("question")

    assert vector == [0.5, 0.6]
    embedding.assert_awaited_once_with(
        model="hosted/test-embedding",
        input=["question"],
    )


def test_embedding_adapter_satisfies_provider_contract() -> None:
    """RAG code can depend on the protocol instead of the LiteLLM implementation."""
    provider: EmbeddingProvider = LiteLLMEmbeddingProvider("hosted/test-embedding")

    assert provider is not None

