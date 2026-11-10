"""FastAPI dependencies that compose application interfaces with adapters."""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from langgraph.checkpoint.base import BaseCheckpointSaver

from insightflow.core.config import Settings, get_settings
from insightflow.core.exceptions import ProviderConfigurationError
from insightflow.providers.embeddings import EmbeddingProvider, LiteLLMEmbeddingProvider
from insightflow.providers.llm import ChatProvider, LiteLLMChatProvider
from insightflow.rag.protocols import ChunkSearcher, ContextAssembler, Retriever
from insightflow.rag.retrieval import BudgetedContextAssembler, DenseRetriever
from insightflow.storage.qdrant import QdrantStore
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
