"""FastAPI dependencies that compose application interfaces with adapters."""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends

from insightflow.core.config import Settings, get_settings
from insightflow.core.exceptions import ProviderConfigurationError
from insightflow.providers.embeddings import EmbeddingProvider, LiteLLMEmbeddingProvider
from insightflow.providers.llm import ChatProvider, LiteLLMChatProvider
from insightflow.rag.protocols import ChunkSearcher, ContextAssembler, Retriever
from insightflow.rag.retrieval import BudgetedContextAssembler, DenseRetriever
from insightflow.storage.qdrant import QdrantStore

SettingsDependency = Annotated[Settings, Depends(get_settings)]


def get_chat_provider(settings: SettingsDependency) -> ChatProvider:
    """Build the configured chat provider behind its application protocol."""
    return LiteLLMChatProvider(
        model=settings.litellm_chat_model,
        api_key=settings.litellm_api_key,
        api_base=settings.litellm_api_base,
    )


def get_embedding_provider(settings: SettingsDependency) -> EmbeddingProvider:
    """Build the configured embedding provider behind its application protocol."""
    if not settings.litellm_embedding_model.strip():
        raise ProviderConfigurationError
    return LiteLLMEmbeddingProvider(
        model=settings.litellm_embedding_model,
        api_key=settings.litellm_api_key,
        api_base=settings.litellm_api_base,
    )


async def get_chunk_searcher(settings: SettingsDependency) -> AsyncIterator[ChunkSearcher]:
    """Open one Qdrant client for the request and close it afterwards."""
    store = QdrantStore(settings)
    try:
        yield store
    finally:
        await store.close()


ChatProviderDependency = Annotated[ChatProvider, Depends(get_chat_provider)]
EmbeddingProviderDependency = Annotated[EmbeddingProvider, Depends(get_embedding_provider)]


def get_retriever(
    embedding_provider: EmbeddingProviderDependency,
    searcher: Annotated[ChunkSearcher, Depends(get_chunk_searcher)],
) -> Retriever:
    """Compose dense retrieval from the embedding provider and the vector store."""
    return DenseRetriever(embedding_provider, searcher)


def get_context_assembler(embedding_provider: EmbeddingProviderDependency) -> ContextAssembler:
    """Budget context with the embedding model's tokenizer as an approximation."""
    return BudgetedContextAssembler(embedding_provider.count_tokens)


RetrieverDependency = Annotated[Retriever, Depends(get_retriever)]
ContextAssemblerDependency = Annotated[ContextAssembler, Depends(get_context_assembler)]
