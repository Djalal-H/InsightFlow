"""Build the retriever that a `RetrievalConfig` describes.

The order is: dense or hybrid search → query expansion (fusion) → reranking → corrective
grading → neighbor expansion → context compression. Reranking comes before grading so that the
grader reads fewer passages; expansion and compression come last so that they run only on
passages that can enter the context.
"""

from collections.abc import Callable

from insightflow.providers.embeddings import EmbeddingProvider
from insightflow.providers.llm import ChatProvider
from insightflow.providers.rerank import RerankProvider
from insightflow.rag.config import RetrievalConfig
from insightflow.rag.corrective import CorrectiveRetriever
from insightflow.rag.protocols import CandidateProcessor, ChunkReader, ChunkSearcher, Retriever
from insightflow.rag.query_expansion import QueryExpandingRetriever
from insightflow.rag.refinement import (
    ContextCompressor,
    NeighborExpander,
    RefiningRetriever,
    Reranker,
)
from insightflow.rag.retrieval import DenseRetriever, HybridRetriever
from insightflow.rag.sparse import Bm25SparseEncoder


def build_retriever(
    config: RetrievalConfig,
    *,
    embedding_provider: EmbeddingProvider,
    searcher: ChunkSearcher,
    reader: ChunkReader | None = None,
    chat_provider: ChatProvider | None = None,
    rerank_provider: RerankProvider | None = None,
    count_tokens: Callable[[str], int] | None = None,
) -> Retriever:
    """Compose the configured retrieval steps; a missing dependency of a step is an error."""
    retriever: Retriever
    if config.mode == "hybrid":
        retriever = HybridRetriever(
            embedding_provider,
            searcher,
            Bm25SparseEncoder(),
            score_threshold=config.score_threshold,
        )
    else:
        retriever = DenseRetriever(
            embedding_provider, searcher, score_threshold=config.score_threshold
        )

    if config.query_expansion != "off":
        retriever = QueryExpandingRetriever(
            retriever,
            _required(chat_provider, "query expansion needs a chat provider"),
            mode=config.query_expansion,
            max_queries=config.query_expansion_max_queries,
        )
    if config.rerank:
        reranker = Reranker(
            _required(rerank_provider, "reranking needs a rerank provider"),
            top_n=config.rerank_top_n,
        )
        retriever = RefiningRetriever(retriever, [reranker])
    if config.corrective:
        retriever = CorrectiveRetriever(
            retriever, _required(chat_provider, "corrective retrieval needs a chat provider")
        )

    processors: list[CandidateProcessor] = []
    if config.neighbors_per_hit > 0:
        processors.append(
            NeighborExpander(
                _required(reader, "neighbor expansion needs a chunk reader"),
                neighbors=config.neighbors_per_hit,
                limit=config.generation_max_chunks,
            )
        )
    if config.compress_context:
        processors.append(
            ContextCompressor(
                _required(chat_provider, "context compression needs a chat provider"),
                count_tokens or embedding_provider.count_tokens,
                limit=config.generation_max_chunks,
            )
        )
    return RefiningRetriever(retriever, processors) if processors else retriever


def _required[T](value: T | None, message: str) -> T:
    if value is None:
        raise ValueError(message)
    return value
