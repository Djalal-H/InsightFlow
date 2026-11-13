"""FastAPI dependencies that compose application interfaces with adapters."""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from langgraph.checkpoint.base import BaseCheckpointSaver

from insightflow.agents.memory import ConversationMemory
from insightflow.core.config import Settings, get_settings
from insightflow.core.exceptions import ProviderConfigurationError
from insightflow.providers.embeddings import EmbeddingProvider, LiteLLMEmbeddingProvider
from insightflow.providers.llm import ChatProvider, LiteLLMChatProvider
from insightflow.providers.rerank import LiteLLMRerankProvider, RerankProvider
from insightflow.rag.composition import build_retriever
from insightflow.rag.config import RetrievalConfig
from insightflow.rag.protocols import ContextAssembler, Retriever
from insightflow.rag.retrieval import BudgetedContextAssembler
from insightflow.storage.qdrant import QdrantStore
from insightflow.storage.qdrant_memory import QdrantMemoryStore
from insightflow.tools.calculator_server import SERVER_NAME as CALCULATOR_SERVER
from insightflow.tools.calculator_server import TOOL_NAME as CALCULATOR_TOOL
from insightflow.tools.calculator_server import create_calculator_server
from insightflow.tools.mcp_client import McpToolClient
from insightflow.tools.registry import ToolRegistration, ToolSet
from insightflow.tools.web_search import create_tavily_registration

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
        query_input_type=settings.embedding_query_input_type,
        document_input_type=settings.embedding_document_input_type,
    )


def get_rerank_provider(settings: SettingsDependency) -> RerankProvider | None:
    """Build the hosted reranker, or `None` when `LITELLM_RERANK_MODEL` is not set."""
    if not settings.litellm_rerank_model.strip():
        return None
    return LiteLLMRerankProvider(
        model=settings.litellm_rerank_model,
        api_key=settings.litellm_rerank_api_key or settings.litellm_api_key,
        api_base=settings.litellm_rerank_api_base,
    )


async def get_chunk_store(settings: SettingsDependency) -> AsyncIterator[QdrantStore]:
    """Open one Qdrant client for the request and close it afterwards."""
    store = QdrantStore(settings)
    try:
        yield store
    finally:
        await store.close()


def retrieval_config_from_settings(settings: Settings) -> RetrievalConfig:
    """Translate the `RAG_*` retrieval settings; a rerank model turns reranking on."""
    return RetrievalConfig(
        mode=settings.rag_retrieval_mode,
        candidate_top_k=settings.rag_candidate_top_k,
        generation_max_chunks=settings.rag_generation_max_chunks,
        context_token_budget=settings.rag_context_token_budget,
        score_threshold=settings.rag_score_threshold,
        query_expansion=settings.rag_query_expansion,
        query_expansion_max_queries=settings.rag_query_expansion_max_queries,
        rerank=bool(settings.litellm_rerank_model.strip()),
        rerank_top_n=settings.rag_rerank_top_n,
        corrective=settings.rag_corrective_retrieval_enabled,
        neighbors_per_hit=settings.rag_neighbors_per_hit,
        compress_context=settings.rag_context_compression_enabled,
    )


ChatProviderDependency = Annotated[ChatProvider, Depends(get_chat_provider)]
EmbeddingProviderDependency = Annotated[EmbeddingProvider, Depends(get_embedding_provider)]


def get_retriever(
    settings: SettingsDependency,
    embedding_provider: EmbeddingProviderDependency,
    store: Annotated[QdrantStore, Depends(get_chunk_store)],
    rerank_provider: Annotated[RerankProvider | None, Depends(get_rerank_provider)],
) -> Retriever:
    """Compose the configured retrieval steps over the embedding provider and the store.

    The chat provider is built only when a configured step needs it.
    """
    config = retrieval_config_from_settings(settings)
    needs_chat = config.query_expansion != "off" or config.corrective or config.compress_context
    return build_retriever(
        config,
        embedding_provider=embedding_provider,
        searcher=store,
        reader=store,
        chat_provider=get_chat_provider(settings) if needs_chat else None,
        rerank_provider=rerank_provider,
    )


async def get_conversation_memory(
    settings: SettingsDependency,
) -> AsyncIterator[ConversationMemory | None]:
    """Open conversation memory for the request, or give `None` when it is disabled."""
    if not settings.memory_enabled:
        yield None
        return
    store = QdrantMemoryStore(settings)
    try:
        yield ConversationMemory(
            get_embedding_provider(settings),
            store,
            top_k=settings.memory_top_k,
            score_threshold=settings.memory_score_threshold,
        )
    finally:
        await store.close()


def get_context_assembler(embedding_provider: EmbeddingProviderDependency) -> ContextAssembler:
    """Budget context with the embedding model's tokenizer as an approximation."""
    return BudgetedContextAssembler(embedding_provider.count_tokens)


RetrieverDependency = Annotated[Retriever, Depends(get_retriever)]
ContextAssemblerDependency = Annotated[ContextAssembler, Depends(get_context_assembler)]
ConversationMemoryDependency = Annotated[
    ConversationMemory | None, Depends(get_conversation_memory)
]


def get_checkpointer(request: Request) -> BaseCheckpointSaver[str] | None:
    """Return the application checkpointer opened at startup, or `None` when disabled.

    The checkpointer is opened by the application lifespan. Without a lifespan (for example in
    tests that use `httpx.ASGITransport`), requests are stateless unless this is overridden.
    """
    checkpointer: BaseCheckpointSaver[str] | None = getattr(
        request.app.state, "checkpointer", None
    )
    return checkpointer


CheckpointerDependency = Annotated[
    BaseCheckpointSaver[str] | None, Depends(get_checkpointer)
]


def create_tool_set(settings: Settings) -> ToolSet:
    """Build the enabled tool servers. Call it after logging is configured (see `main.py`)."""
    registrations: list[ToolRegistration] = []
    if settings.tool_calculator_enabled:
        calculator = McpToolClient(
            create_calculator_server(),
            server_name=CALCULATOR_SERVER,
            allowed_tools={CALCULATOR_TOOL},
            timeout_seconds=settings.tool_timeout_seconds,
            max_output_chars=settings.tool_max_output_chars,
        )
        registrations.append(ToolRegistration(client=calculator))
    if settings.tool_web_search_enabled and settings.tavily_api_key is not None:
        registrations.append(
            create_tavily_registration(
                settings.tavily_mcp_url,
                settings.tavily_api_key.get_secret_value(),
                max_results=settings.web_search_max_results,
                timeout_seconds=settings.tool_timeout_seconds,
                max_output_chars=settings.tool_max_output_chars,
            )
        )
    return ToolSet(registrations, failure_cooldown_seconds=settings.tool_failure_cooldown_seconds)


def get_tool_set(request: Request) -> ToolSet:
    """Return the tool set built at startup, or an empty one.

    The tool set lives for the whole application, so each server's tool list is fetched once.
    Without a lifespan (for example in tests that use `httpx.ASGITransport`), no tools are
    offered unless this is overridden.
    """
    tools: ToolSet | None = getattr(request.app.state, "tool_set", None)
    return tools if tools is not None else ToolSet()


ToolSetDependency = Annotated[ToolSet, Depends(get_tool_set)]
