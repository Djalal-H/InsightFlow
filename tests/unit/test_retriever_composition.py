"""Tests for building the configured retriever and for the settings that select it."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest

from insightflow.api.dependencies import (
    get_rerank_provider,
    get_retriever,
    retrieval_config_from_settings,
)
from insightflow.core.config import Settings
from insightflow.providers.rerank import LiteLLMRerankProvider
from insightflow.rag.composition import build_retriever
from insightflow.rag.config import RetrievalConfig
from insightflow.rag.corrective import CorrectiveRetriever
from insightflow.rag.models import Chunk, RetrievalQuery, RetrievedChunk, SparseVector
from insightflow.rag.query_expansion import QueryExpandingRetriever
from insightflow.rag.refinement import RefiningRetriever
from insightflow.rag.retrieval import DenseRetriever, HybridRetriever
from tests.fakes import FakeRerankProvider, ScriptedChatProvider, make_candidate

LONG = "The river is ten metres deep. It was mapped in 1990. Many fish live there."


class FakeEmbeddingProvider:
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def count_tokens(self, text: str) -> int:
        return max(1, len(text.split()))


class FakeStore:
    """A searcher and reader over fixed candidates; records every search."""

    def __init__(self, candidates: list[RetrievedChunk]) -> None:
        self.candidates = candidates
        self.searches: list[dict[str, Any]] = []

    async def search_chunks(
        self,
        vector: Sequence[float],
        *,
        limit: int,
        document_ids: Sequence[str] | None = None,
        mime_types: Sequence[str] | None = None,
        source_identifiers: Sequence[str] | None = None,
        score_threshold: float | None = None,
        sparse_vector: SparseVector | None = None,
    ) -> list[RetrievedChunk]:
        self.searches.append({"sparse": sparse_vector, "threshold": score_threshold})
        return self.candidates[:limit]

    async def get_chunks(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        return [
            make_candidate(chunk_id, f"Neighbor {chunk_id}.").chunk for chunk_id in chunk_ids
        ]


def build(config: RetrievalConfig, store: FakeStore, **dependencies: Any) -> Any:
    return build_retriever(
        config, embedding_provider=FakeEmbeddingProvider(), searcher=store, **dependencies
    )


def test_default_config_builds_plain_dense_retrieval() -> None:
    retriever = build(RetrievalConfig(), FakeStore([]))

    assert type(retriever) is DenseRetriever


async def test_hybrid_mode_sends_a_sparse_query_and_the_threshold() -> None:
    store = FakeStore([make_candidate()])
    retriever = build(RetrievalConfig(mode="hybrid", score_threshold=0.4), store)

    await retriever.retrieve(RetrievalQuery(text="river depth"))

    assert isinstance(retriever, HybridRetriever)
    assert len(store.searches[0]["sparse"].indices) == 2
    assert store.searches[0]["threshold"] == 0.4


async def test_every_step_runs_in_the_documented_order() -> None:
    candidates = [
        make_candidate("a", "The desert is dry.", rank=1),
        make_candidate("b", LONG, rank=2, next_chunk_id="b-next"),
        make_candidate("c", "A river bank.", rank=3),
    ]
    store = FakeStore(candidates)
    chat = ScriptedChatProvider(
        expand=json.dumps({"queries": ["river depth in metres"]}),
        grade=json.dumps({"relevant": [1]}),
        compress=json.dumps({"keep": {"1": [1]}}),
    )
    rerank = FakeRerankProvider("river")
    config = RetrievalConfig(
        mode="hybrid",
        query_expansion="multi_query",
        rerank=True,
        rerank_top_n=2,
        corrective=True,
        neighbors_per_hit=1,
        compress_context=True,
    )

    retriever = build(
        config, store, reader=store, chat_provider=chat, rerank_provider=rerank
    )
    results = await retriever.retrieve(RetrievalQuery(text="How deep is the river?"))

    assert chat.roles() == ["expand", "grade", "compress"]
    assert len(store.searches) == 2  # the question and one more wording
    assert len(rerank.calls) == 1 and rerank.calls[0][2] == 2
    # Reranking kept b and c, grading kept b, expansion added its neighbor, compression cut it.
    [result] = results
    assert result.chunk.chunk_id == "b"
    assert result.chunk.metadata["expanded_chunk_ids"] == ["b-next"]
    assert result.chunk.content == "The river is ten metres deep. [...]"


def test_wrappers_are_layered_from_search_to_context() -> None:
    store = FakeStore([])
    chat = ScriptedChatProvider()

    expanded = build(RetrievalConfig(query_expansion="decompose"), store, chat_provider=chat)
    corrective = build(RetrievalConfig(corrective=True), store, chat_provider=chat)
    expanding = build(RetrievalConfig(neighbors_per_hit=2), store, reader=store)

    assert isinstance(expanded, QueryExpandingRetriever)
    assert isinstance(corrective, CorrectiveRetriever)
    assert isinstance(expanding, RefiningRetriever)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (RetrievalConfig(query_expansion="multi_query"), "query expansion needs"),
        (RetrievalConfig(rerank=True), "reranking needs"),
        (RetrievalConfig(corrective=True), "corrective retrieval needs"),
        (RetrievalConfig(neighbors_per_hit=1), "neighbor expansion needs"),
        (RetrievalConfig(compress_context=True), "context compression needs"),
    ],
)
def test_a_step_without_its_dependency_is_refused(config: RetrievalConfig, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build(config, FakeStore([]))


# --- settings ----------------------------------------------------------------------------


def test_default_settings_keep_every_stage_6_option_off() -> None:
    settings = Settings(_env_file=None, litellm_rerank_model="", rag_retrieval_mode="dense")

    config = retrieval_config_from_settings(
        settings.model_copy(
            update={
                "rag_query_expansion": "off",
                "rag_corrective_retrieval_enabled": False,
                "rag_neighbors_per_hit": 0,
                "rag_context_compression_enabled": False,
            }
        )
    )

    assert config == RetrievalConfig(
        candidate_top_k=settings.rag_candidate_top_k,
        generation_max_chunks=settings.rag_generation_max_chunks,
        context_token_budget=settings.rag_context_token_budget,
        score_threshold=settings.rag_score_threshold,
        query_expansion_max_queries=settings.rag_query_expansion_max_queries,
        rerank_top_n=settings.rag_rerank_top_n,
    )
    for name in (
        "rag_retrieval_mode",
        "rag_query_expansion",
        "litellm_rerank_model",
        "rag_corrective_retrieval_enabled",
        "rag_neighbors_per_hit",
        "rag_context_compression_enabled",
        "rag_search_filters_enabled",
        "agent_multi_hop_enabled",
        "memory_enabled",
    ):
        assert Settings.model_fields[name].default in ("dense", "off", "", False, 0)


def test_settings_select_each_retrieval_step() -> None:
    settings = Settings(
        _env_file=None,
        rag_retrieval_mode="hybrid",
        rag_score_threshold=0.2,
        rag_query_expansion="decompose",
        rag_query_expansion_max_queries=4,
        litellm_rerank_model="cohere/rerank-test",
        rag_rerank_top_n=5,
        rag_corrective_retrieval_enabled=True,
        rag_neighbors_per_hit=2,
        rag_context_compression_enabled=True,
    )

    config = retrieval_config_from_settings(settings)

    assert config.mode == "hybrid"
    assert config.score_threshold == 0.2
    assert (config.query_expansion, config.query_expansion_max_queries) == ("decompose", 4)
    assert (config.rerank, config.rerank_top_n) == (True, 5)
    assert config.corrective and config.compress_context
    assert config.neighbors_per_hit == 2


def test_rerank_provider_is_built_only_with_a_model() -> None:
    without = Settings(_env_file=None, litellm_rerank_model=" ")
    with_model = Settings(
        _env_file=None,
        litellm_rerank_model="cohere/rerank-test",
        litellm_api_key="chat-key",
        litellm_rerank_api_base="https://rerank.test",
    )

    provider = get_rerank_provider(with_model)

    assert get_rerank_provider(without) is None
    assert isinstance(provider, LiteLLMRerankProvider)
    assert provider._api_key == "chat-key"
    assert provider._api_base == "https://rerank.test"


def test_get_retriever_builds_the_chat_provider_only_when_a_step_needs_it() -> None:
    store: Any = FakeStore([])
    plain = Settings(
        _env_file=None,
        litellm_chat_model="",
        litellm_rerank_model="",
        rag_retrieval_mode="dense",
        rag_query_expansion="off",
        rag_corrective_retrieval_enabled=False,
        rag_context_compression_enabled=False,
        rag_neighbors_per_hit=0,
    )
    with_chat = plain.model_copy(
        update={"litellm_chat_model": "openai/test", "rag_corrective_retrieval_enabled": True}
    )

    assert type(get_retriever(plain, FakeEmbeddingProvider(), store, None)) is DenseRetriever
    assert isinstance(
        get_retriever(with_chat, FakeEmbeddingProvider(), store, None), CorrectiveRetriever
    )
