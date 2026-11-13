"""Dense and hybrid retrieval, and budgeted context assembly over stored chunks."""

from collections.abc import Callable

from insightflow.providers.embeddings import EmbeddingProvider
from insightflow.rag.models import (
    AssembledContext,
    ContextBudget,
    RetrievalQuery,
    RetrievedChunk,
    SparseVector,
)
from insightflow.rag.protocols import ChunkSearcher, ContextAssembler, Retriever, SparseEncoder


class DenseRetriever:
    """Embed the query text and rank stored chunks by vector similarity."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        searcher: ChunkSearcher,
        *,
        score_threshold: float | None = None,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._searcher = searcher
        self._score_threshold = score_threshold

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        """Return up to `top_k` chunks, honoring document, source, and source-type filters."""
        if query.languages or query.date_from or query.date_to:
            raise ValueError("language and date filters are not supported by stored chunks")
        vector = await self._embedding_provider.embed_query(query.text)
        return await self._searcher.search_chunks(
            vector,
            limit=query.top_k,
            document_ids=query.document_ids,
            mime_types=query.source_types,
            source_identifiers=query.source_identifiers,
            score_threshold=self._score_threshold,
            sparse_vector=self._sparse_vector(query.text),
        )

    def _sparse_vector(self, text: str) -> SparseVector | None:
        return None


class HybridRetriever(DenseRetriever):
    """Fuse the dense ranking with a lexical ranking over stored sparse vectors.

    The collection must hold sparse vectors written by the same encoder (ingest with
    `RAG_RETRIEVAL_MODE=hybrid`). The scores are fusion scores, not similarities.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        searcher: ChunkSearcher,
        sparse_encoder: SparseEncoder,
        *,
        score_threshold: float | None = None,
    ) -> None:
        super().__init__(embedding_provider, searcher, score_threshold=score_threshold)
        self._sparse_encoder = sparse_encoder

    def _sparse_vector(self, text: str) -> SparseVector | None:
        return self._sparse_encoder.encode_query(text)


def renumber(candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Return the candidates in their given order with ranks 1..n."""
    return [
        candidate if candidate.rank == rank else candidate.model_copy(update={"rank": rank})
        for rank, candidate in enumerate(candidates, start=1)
    ]


def render_context_block(citation: int, candidate: RetrievedChunk) -> str:
    """Render one numbered passage with the provenance a reader can cite."""
    chunk = candidate.chunk
    header = f"[{citation}] {candidate.source_identifier}"
    if chunk.page_start is not None:
        if chunk.page_start == chunk.page_end:
            header += f" (page {chunk.page_start})"
        else:
            header += f" (pages {chunk.page_start}-{chunk.page_end})"
    if chunk.heading_path:
        header += f"\nSection: {' > '.join(chunk.heading_path)}"
    return f"{header}\n{chunk.content}"


class BudgetedContextAssembler:
    """Keep the best-ranked passages that fit the chunk and token limits."""

    def __init__(self, count_tokens: Callable[[str], int]) -> None:
        self._count_tokens = count_tokens

    def assemble(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        budget: ContextBudget,
    ) -> AssembledContext:
        """Select candidates in rank order; a passage that does not fit is excluded."""
        selected: list[RetrievedChunk] = []
        blocks: list[str] = []
        excluded: list[str] = []
        token_count = 0
        for candidate in sorted(candidates, key=lambda item: item.rank):
            if len(selected) >= budget.max_chunks:
                excluded.append(candidate.chunk.chunk_id)
                continue
            block = render_context_block(len(selected) + 1, candidate)
            block_tokens = self._count_tokens(block)
            if token_count + block_tokens > budget.max_tokens:
                excluded.append(candidate.chunk.chunk_id)
                continue
            selected.append(candidate)
            blocks.append(block)
            token_count += block_tokens
        return AssembledContext(
            query=query,
            chunks=selected,
            rendered="\n\n".join(blocks),
            token_count=token_count,
            excluded_chunk_ids=excluded,
        )


__all__ = [
    "AssembledContext",
    "BudgetedContextAssembler",
    "ContextAssembler",
    "ContextBudget",
    "DenseRetriever",
    "HybridRetriever",
    "RetrievalQuery",
    "RetrievedChunk",
    "Retriever",
    "render_context_block",
    "renumber",
]
