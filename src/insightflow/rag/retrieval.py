"""Dense retrieval and budgeted context assembly over stored chunks."""

from collections.abc import Callable

from insightflow.providers.embeddings import EmbeddingProvider
from insightflow.rag.models import (
    AssembledContext,
    ContextBudget,
    RetrievalQuery,
    RetrievedChunk,
)
from insightflow.rag.protocols import ChunkSearcher, ContextAssembler, Retriever


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
        """Return up to `top_k` chunks, honoring document and source-type filters."""
        if query.languages or query.date_from or query.date_to:
            raise ValueError("language and date filters are not supported by stored chunks")
        vector = await self._embedding_provider.embed_query(query.text)
        return await self._searcher.search_chunks(
            vector,
            limit=query.top_k,
            document_ids=query.document_ids,
            mime_types=query.source_types,
            score_threshold=self._score_threshold,
        )


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
    "RetrievalQuery",
    "RetrievedChunk",
    "Retriever",
    "render_context_block",
]
