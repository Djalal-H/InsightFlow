"""Retrieval-based conversation memory: earlier turns of a thread, found by similarity.

The checkpoint keeps the whole history, but prompts show only the last turns. This memory
embeds each completed turn and, for a new question, finds the older turns of the same thread
that are most similar to it, so a long conversation can still use what was said early.
"""

from collections.abc import Sequence
from typing import Protocol

from pydantic import Field

from insightflow.agents.models import ConversationTurn, WorkflowModel
from insightflow.providers.embeddings import EmbeddingProvider

MEMORY_TEXT_MAX_CHARS = 4000


class RememberedTurn(WorkflowModel):
    """One stored turn with its position in the thread and its similarity to a query."""

    turn_index: int = Field(ge=0)
    query: str
    answer: str
    score: float = 0.0


class MemoryStore(Protocol):
    """Persist and search embedded conversation turns, separated by thread."""

    async def save_turn(
        self,
        thread_id: str,
        turn_index: int,
        vector: Sequence[float],
        *,
        query: str,
        answer: str,
    ) -> None: ...

    async def search_turns(
        self,
        thread_id: str,
        vector: Sequence[float],
        *,
        limit: int,
        before_turn: int,
        score_threshold: float | None = None,
    ) -> list[RememberedTurn]: ...

    async def delete_thread(self, thread_id: str) -> None: ...


def memory_text(query: str, answer: str) -> str:
    """The text that is embedded for one turn."""
    return f"User: {query}\nAssistant: {answer}"[:MEMORY_TEXT_MAX_CHARS]


class ConversationMemory:
    """Remember completed turns and recall those that are relevant to a new question."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        store: MemoryStore,
        *,
        top_k: int = 3,
        score_threshold: float | None = None,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self._embedding_provider = embedding_provider
        self._store = store
        self._top_k = top_k
        self._score_threshold = score_threshold

    async def remember(self, thread_id: str, turn_index: int, turn: ConversationTurn) -> None:
        """Store one completed turn; storing the same position again replaces it."""
        if not turn.answer.strip():
            return
        vectors = await self._embedding_provider.embed_documents(
            [memory_text(turn.query, turn.answer)]
        )
        await self._store.save_turn(
            thread_id, turn_index, vectors[0], query=turn.query, answer=turn.answer
        )

    async def recall(
        self, thread_id: str, query: str, *, before_turn: int
    ) -> list[ConversationTurn]:
        """Return the most similar turns with a position below `before_turn`, oldest first."""
        if before_turn <= 0:
            return []
        vector = await self._embedding_provider.embed_query(query)
        found = await self._store.search_turns(
            thread_id,
            vector,
            limit=self._top_k,
            before_turn=before_turn,
            score_threshold=self._score_threshold,
        )
        return [
            ConversationTurn(query=turn.query, answer=turn.answer)
            for turn in sorted(found, key=lambda turn: turn.turn_index)
        ]

    async def forget(self, thread_id: str) -> None:
        """Delete every stored turn of a thread."""
        await self._store.delete_thread(thread_id)
