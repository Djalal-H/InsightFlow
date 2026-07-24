"""Tests for provider construction at the application composition boundary."""

from unittest.mock import patch

import pytest

from insightflow.api.dependencies import get_chat_provider
from insightflow.core.config import Settings
from insightflow.providers.llm import LiteLLMChatProvider


@pytest.mark.parametrize(
    "model",
    [
        "openai/gpt-test",
        "anthropic/claude-test",
    ],
)
def test_chat_provider_selection_comes_from_settings(model: str) -> None:
    """Changing provider prefixes requires settings changes, not route branches."""
    settings = Settings(
        _env_file=None,
        litellm_chat_model=model,
        litellm_api_key="test-key",
        litellm_api_base="https://provider.test/v1",
    )

    with patch(
        "insightflow.api.dependencies.LiteLLMChatProvider",
        autospec=True,
    ) as provider_class:
        provider = get_chat_provider(settings)

    assert provider is provider_class.return_value
    provider_class.assert_called_once_with(
        model=model,
        api_key="test-key",
        api_base="https://provider.test/v1",
    )


def test_chat_provider_dependency_returns_protocol_implementation() -> None:
    """The default dependency composes the LiteLLM implementation behind the protocol."""
    settings = Settings(
        _env_file=None,
        litellm_chat_model="openai/test-model",
    )

    with patch(
        "insightflow.providers.llm.get_llm_provider",
        return_value=("test-model", "openai", None, None),
    ):
        provider = get_chat_provider(settings)

    assert isinstance(provider, LiteLLMChatProvider)
