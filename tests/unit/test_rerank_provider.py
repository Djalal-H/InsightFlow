"""Tests for the LiteLLM rerank adapter with mocked provider calls."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from litellm.exceptions import AuthenticationError, RateLimitError

from insightflow.core.exceptions import (
    ProviderConfigurationError,
    ProviderError,
    ProviderRateLimitError,
)
from insightflow.providers.rerank import LiteLLMRerankProvider, RerankResult


def response(*results: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(results=list(results))


async def test_rerank_sends_documents_and_orders_results_best_first() -> None:
    provider = LiteLLMRerankProvider(
        " cohere/rerank-test ", api_key="key", api_base="https://rerank.test"
    )
    reply = response(
        {"index": 0, "relevance_score": 0.1},
        {"index": 2, "relevance_score": 0.9},
    )

    with patch("insightflow.providers.rerank.arerank", AsyncMock(return_value=reply)) as call:
        results = await provider.rerank("q", ["a", "b", "c"], top_n=5)

    assert results == [RerankResult(index=2, score=0.9), RerankResult(index=0, score=0.1)]
    call.assert_awaited_once_with(
        model="cohere/rerank-test",
        query="q",
        documents=["a", "b", "c"],
        top_n=3,
        return_documents=False,
        api_key="key",
        api_base="https://rerank.test",
    )


async def test_rerank_without_documents_makes_no_request() -> None:
    with patch("insightflow.providers.rerank.arerank", AsyncMock()) as call:
        assert await LiteLLMRerankProvider("cohere/rerank-test").rerank("q", [], top_n=3) == []

    call.assert_not_awaited()


@pytest.mark.parametrize(
    "reply",
    [
        SimpleNamespace(results=None),
        response({"index": 5, "relevance_score": 0.5}),
        response({"index": 0, "relevance_score": 0.5}, {"index": 0, "relevance_score": 0.4}),
        response({"index": 0}),
        response({"index": 0, "relevance_score": float("nan")}),
        response({"index": True, "relevance_score": 0.5}),
    ],
)
async def test_invalid_rerank_responses_are_provider_errors(reply: SimpleNamespace) -> None:
    with patch("insightflow.providers.rerank.arerank", AsyncMock(return_value=reply)):
        with pytest.raises(ProviderError, match="invalid response"):
            await LiteLLMRerankProvider("cohere/rerank-test").rerank("q", ["a"], top_n=1)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (RateLimitError("limited", "cohere", "rerank-test"), ProviderRateLimitError),
        (AuthenticationError("denied", "cohere", "rerank-test"), ProviderConfigurationError),
    ],
)
async def test_rerank_maps_provider_failures(
    error: Exception, expected: type[Exception]
) -> None:
    with patch("insightflow.providers.rerank.arerank", AsyncMock(side_effect=error)):
        with pytest.raises(expected):
            await LiteLLMRerankProvider("cohere/rerank-test").rerank("q", ["a"], top_n=1)


def test_rerank_provider_requires_a_model() -> None:
    with pytest.raises(ProviderConfigurationError):
        LiteLLMRerankProvider("  ")
