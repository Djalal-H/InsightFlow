"""Corrective retrieval: grade the passages, drop the irrelevant, and search once more.

A chat model reads the retrieved passages and lists those that help to answer the question.
When none helps, it writes a different query; that query is searched once, and its passages
are graded against the original question. No relevant passage means an empty result, which
tells the agent that the documents did not answer this search. Unreadable grader output keeps
the passages as retrieved.
"""

from pydantic import BaseModel, ConfigDict, Field

from insightflow.providers.llm import ChatProvider
from insightflow.providers.structured import complete_json
from insightflow.rag.models import RetrievalQuery, RetrievedChunk
from insightflow.rag.protocols import Retriever
from insightflow.rag.retrieval import render_context_block, renumber

MAX_QUERY_CHARS = 500
GRADER_SYSTEM_PROMPT = (
    "You grade search results for a question. Return only a JSON object: "
    '{"relevant": [<passage numbers>], "query": "<search query or empty>"}. '
    "List the number of each passage that helps to answer the question, also when it gives "
    "only a part of the answer. If no passage helps, write in query a different search query "
    "that can find the answer in the documents. If some passage helps, leave query empty. "
    "Do not answer the question."
)


class RelevanceGrade(BaseModel):
    """Chat-model output for passage grading."""

    model_config = ConfigDict(extra="ignore")

    relevant: list[int] = Field(default_factory=list)
    query: str = ""


class CorrectiveRetriever:
    """Keep the passages that a grader finds relevant; retry once with its better query."""

    def __init__(self, inner: Retriever, chat_provider: ChatProvider) -> None:
        self._inner = inner
        self._chat_provider = chat_provider

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        candidates = await self._inner.retrieve(query)
        if not candidates:
            return []
        grade = await self._grade(query.text, candidates)
        if grade is None:
            return candidates
        kept = _relevant(candidates, grade)
        better_query = grade.query.strip()[:MAX_QUERY_CHARS]
        if kept or not better_query or better_query.casefold() == query.text.strip().casefold():
            return kept

        retried = await self._inner.retrieve(query.model_copy(update={"text": better_query}))
        if not retried:
            return []
        grade = await self._grade(query.text, retried)
        return retried if grade is None else _relevant(retried, grade)

    async def _grade(
        self, question: str, candidates: list[RetrievedChunk]
    ) -> RelevanceGrade | None:
        passages = "\n\n".join(
            render_context_block(number, candidate)
            for number, candidate in enumerate(candidates, start=1)
        )
        messages = [
            {"role": "system", "content": GRADER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nPassages:\n{passages}"},
        ]
        return await complete_json(self._chat_provider, messages, RelevanceGrade)


def _relevant(candidates: list[RetrievedChunk], grade: RelevanceGrade) -> list[RetrievedChunk]:
    numbers = set(grade.relevant)
    return renumber(
        [
            candidate
            for number, candidate in enumerate(candidates, start=1)
            if number in numbers
        ]
    )
