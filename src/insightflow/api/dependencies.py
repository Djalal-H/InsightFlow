"""FastAPI dependencies that compose application interfaces with adapters."""

from typing import Annotated

from fastapi import Depends

from insightflow.core.config import Settings, get_settings
from insightflow.providers.llm import ChatProvider, LiteLLMChatProvider


def get_chat_provider(
    settings: Annotated[Settings, Depends(get_settings)],
) -> ChatProvider:
    """Build the configured chat provider behind its application protocol."""
    return LiteLLMChatProvider(
        model=settings.litellm_chat_model,
        api_key=settings.litellm_api_key,
        api_base=settings.litellm_api_base,
    )


ChatProviderDependency = Annotated[ChatProvider, Depends(get_chat_provider)]
