"""Qdrant adapter for conversation memory: one point per completed turn."""

from collections.abc import Sequence
from uuid import uuid5

from qdrant_client import AsyncQdrantClient, models

from insightflow.agents.memory import RememberedTurn
from insightflow.core.config import Settings
from insightflow.core.exceptions import StorageError
from insightflow.rag.identity import INSIGHTFLOW_RAG_NAMESPACE

VECTOR_NAME = "dense"


def memory_point_id(thread_id: str, turn_index: int) -> str:
    """A deterministic point identifier, so that a turn is stored once."""
    return str(uuid5(INSIGHTFLOW_RAG_NAMESPACE, f"memory:{thread_id}:{turn_index}"))


class QdrantMemoryStore:
    """Store turn vectors in their own collection, created on the first write."""

    def __init__(self, settings: Settings, *, client: AsyncQdrantClient | None = None) -> None:
        self.collection_name = settings.memory_collection
        self._client = client or AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )

    async def save_turn(
        self,
        thread_id: str,
        turn_index: int,
        vector: Sequence[float],
        *,
        query: str,
        answer: str,
    ) -> None:
        try:
            if not await self._client.collection_exists(self.collection_name):
                await self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        VECTOR_NAME: models.VectorParams(
                            size=len(vector), distance=models.Distance.COSINE
                        )
                    },
                )
                await self._client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="thread_id",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            await self._client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=memory_point_id(thread_id, turn_index),
                        vector={VECTOR_NAME: list(vector)},
                        payload={
                            "thread_id": thread_id,
                            "turn_index": turn_index,
                            "query": query,
                            "answer": answer,
                        },
                    )
                ],
                wait=True,
            )
        except Exception as exc:
            raise StorageError("Qdrant memory write failed") from exc

    async def search_turns(
        self,
        thread_id: str,
        vector: Sequence[float],
        *,
        limit: int,
        before_turn: int,
        score_threshold: float | None = None,
    ) -> list[RememberedTurn]:
        try:
            if not await self._client.collection_exists(self.collection_name):
                return []
            result = await self._client.query_points(
                collection_name=self.collection_name,
                query=list(vector),
                using=VECTOR_NAME,
                query_filter=models.Filter(
                    must=[
                        _thread(thread_id),
                        models.FieldCondition(
                            key="turn_index", range=models.Range(lt=before_turn)
                        ),
                    ]
                ),
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
            )
        except Exception as exc:
            raise StorageError("Qdrant memory search failed") from exc
        return [
            RememberedTurn(
                turn_index=payload["turn_index"],
                query=payload["query"],
                answer=payload["answer"],
                score=point.score,
            )
            for point in result.points
            if (payload := point.payload)
        ]

    async def delete_thread(self, thread_id: str) -> None:
        try:
            if not await self._client.collection_exists(self.collection_name):
                return
            await self._client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(must=[_thread(thread_id)])
                ),
                wait=True,
            )
        except Exception as exc:
            raise StorageError("Qdrant memory delete failed") from exc

    async def close(self) -> None:
        """Release client network resources."""
        await self._client.close()


def _thread(thread_id: str) -> models.FieldCondition:
    return models.FieldCondition(key="thread_id", match=models.MatchValue(value=thread_id))
