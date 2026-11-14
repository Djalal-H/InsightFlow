"""Provider-independent reranking contract backed by LiteLLM's hosted rerank APIs."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any, Protocol

from litellm import arerank

from insightflow.core.exceptions import ProviderConfigurationError, ProviderError
from insightflow.providers.llm import translate_provider_errors


@dataclass(frozen=True)
class RerankResult:
    """The relevance of one input document: its input position and the model's score."""

    index: int
    score: float


class RerankProvider(Protocol):
    """Contract used by retrieval to order passages with a hosted reranking model."""

    async def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int
    ) -> list[RerankResult]:
        """Return at most `top_n` results, best first, each for a distinct document."""
        ...


class LiteLLMRerankProvider:
    """Call any LiteLLM-supported hosted reranker (Cohere, NVIDIA NIM, Jina, Voyage, ...)."""

    def __init__(self, model: str, api_key: str | None = None, api_base: str | None = None) -> None:
        normalized_model = model.strip()
        if not normalized_model:
            raise ProviderConfigurationError
        self._model = normalized_model
        self._api_key = api_key
        self._api_base = api_base

    async def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int
    ) -> list[RerankResult]:
        if not documents:
            return []
        options: dict[str, str] = {}
        if self._api_key:
            options["api_key"] = self._api_key
        if self._api_base:
            options["api_base"] = self._api_base
        with translate_provider_errors():
            response = await arerank(
                model=self._model,
                query=query,
                documents=list(documents),
                top_n=min(top_n, len(documents)),
                return_documents=False,
                **options,
            )
        return _parse_results(getattr(response, "results", None), len(documents))[:top_n]


def _parse_results(raw: Any, document_count: int) -> list[RerankResult]:
    """Validate the provider's results and order them best first."""
    if not isinstance(raw, list):
        raise ProviderError("The rerank provider returned an invalid response")
    results: list[RerankResult] = []
    seen: set[int] = set()
    for item in raw:
        try:
            index, score = item["index"], float(item["relevance_score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderError("The rerank provider returned an invalid response") from exc
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < document_count
            or index in seen
            or not isfinite(score)
        ):
            raise ProviderError("The rerank provider returned an invalid response")
        seen.add(index)
        results.append(RerankResult(index=index, score=score))
    return sorted(results, key=lambda result: result.score, reverse=True)
