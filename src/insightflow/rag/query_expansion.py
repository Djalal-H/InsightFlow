"""Query expansion: retrieve for several wordings or sub-questions and fuse the rankings.

`multi_query` asks a chat model for other wordings of the question (query rewriting).
`decompose` asks for one query per fact that the question needs (multi-hop questions whose
parts do not depend on each other). The original query is always searched too, and the searches
run concurrently. Unreadable model output means that only the original query is searched.
"""

import asyncio
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from insightflow.providers.llm import ChatProvider
from insightflow.providers.structured import complete_json
from insightflow.rag.models import RetrievalQuery, RetrievedChunk
from insightflow.rag.protocols import Retriever

ExpansionMode = Literal["multi_query", "decompose"]

MAX_QUERY_CHARS = 500
RRF_K = 60
_JSON_FORMAT = 'Return only a JSON object: {"queries": ["<query>", ...]}. '
MULTI_QUERY_SYSTEM_PROMPT = (
    "You write search queries for a document index. "
    + _JSON_FORMAT
    + "Write other wordings of the question: use synonyms and the terms that a document "
    "would use. Each query must be complete without the others. Do not answer the question."
)
DECOMPOSE_SYSTEM_PROMPT = (
    "You split a question into search queries for a document index. "
    + _JSON_FORMAT
    + "If the question needs several facts, write one query for each fact. Each query must be "
    "complete without the others. If the question needs one fact, return an empty list. "
    "Do not answer the question."
)
_SYSTEM_PROMPTS: dict[ExpansionMode, str] = {
    "multi_query": MULTI_QUERY_SYSTEM_PROMPT,
    "decompose": DECOMPOSE_SYSTEM_PROMPT,
}


class ExpandedQueries(BaseModel):
    """Chat-model output for query expansion."""

    model_config = ConfigDict(extra="ignore")

    queries: list[str] = Field(default_factory=list)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievedChunk]], *, limit: int, k: int = RRF_K
) -> list[RetrievedChunk]:
    """Fuse rankings by the sum of `1 / (k + rank)`; ties keep first-seen order.

    The fused score replaces the candidate's score.
    """
    scores: dict[str, float] = {}
    first_seen: dict[str, RetrievedChunk] = {}
    for ranking in rankings:
        for position, candidate in enumerate(ranking, start=1):
            chunk_id = candidate.chunk.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + position)
            first_seen.setdefault(chunk_id, candidate)
    ordered = sorted(first_seen, key=lambda chunk_id: scores[chunk_id], reverse=True)[:limit]
    return [
        first_seen[chunk_id].model_copy(update={"score": scores[chunk_id], "rank": rank})
        for rank, chunk_id in enumerate(ordered, start=1)
    ]


class QueryExpandingRetriever:
    """Search the original query and model-written queries, then fuse the results."""

    def __init__(
        self,
        inner: Retriever,
        chat_provider: ChatProvider,
        *,
        mode: ExpansionMode,
        max_queries: int = 3,
    ) -> None:
        if max_queries < 2:
            raise ValueError("max_queries must be at least 2")
        self._inner = inner
        self._chat_provider = chat_provider
        self._mode: ExpansionMode = mode
        self._max_queries = max_queries

    async def expand(self, text: str) -> list[str]:
        """Return the original query first, then the distinct model-written queries."""
        extra = self._max_queries - 1
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPTS[self._mode]},
            {"role": "user", "content": f"Question: {text}\n\nWrite at most {extra} queries."},
        ]
        expanded = await complete_json(self._chat_provider, messages, ExpandedQueries)
        queries = [text]
        seen = {text.strip().casefold()}
        for query in expanded.queries if expanded is not None else []:
            cleaned = query.strip()[:MAX_QUERY_CHARS]
            if cleaned and cleaned.casefold() not in seen and len(queries) < self._max_queries:
                seen.add(cleaned.casefold())
                queries.append(cleaned)
        return queries

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        queries = await self.expand(query.text)
        if len(queries) == 1:
            return await self._inner.retrieve(query)
        rankings = await asyncio.gather(
            *(self._inner.retrieve(query.model_copy(update={"text": text})) for text in queries)
        )
        return reciprocal_rank_fusion(rankings, limit=query.top_k)
