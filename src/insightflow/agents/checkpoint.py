"""Checkpointer construction and conversation-thread reads."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import aiosqlite
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel

from insightflow.agents.models import ConversationTurn, VerificationResult
from insightflow.rag.models import AssembledContext

CheckpointBackend = Literal["none", "memory", "sqlite"]

# Pydantic types stored at the top level of the workflow state. Nested models are rebuilt by
# Pydantic validation, so only these need to be allowed.
CHECKPOINT_STATE_TYPES: tuple[type[BaseModel], ...] = (
    AssembledContext,
    ConversationTurn,
    VerificationResult,
)


def create_checkpoint_serializer() -> JsonPlusSerializer:
    """Return a serializer that restores only the workflow's own model types."""
    return JsonPlusSerializer(
        allowed_msgpack_modules=[(cls.__module__, cls.__name__) for cls in CHECKPOINT_STATE_TYPES]
    )


@asynccontextmanager
async def open_checkpointer(
    backend: CheckpointBackend,
    sqlite_path: Path,
) -> AsyncIterator[BaseCheckpointSaver[str] | None]:
    """Open the configured checkpointer for the lifetime of the application."""
    if backend == "none":
        yield None
        return
    if backend == "memory":
        yield InMemorySaver(serde=create_checkpoint_serializer())
        return

    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(sqlite_path) as connection:
        saver = AsyncSqliteSaver(connection, serde=create_checkpoint_serializer())
        await saver.setup()
        yield saver


def thread_config(thread_id: str) -> RunnableConfig:
    """Return the LangGraph run configuration for one conversation thread."""
    return {"configurable": {"thread_id": thread_id}}


async def read_thread_history(
    checkpointer: BaseCheckpointSaver[str],
    thread_id: str,
) -> list[ConversationTurn] | None:
    """Return a thread's completed turns, or `None` when the thread has no checkpoint."""
    checkpoint = await checkpointer.aget_tuple(thread_config(thread_id))
    if checkpoint is None:
        return None
    history = checkpoint.checkpoint["channel_values"].get("history", [])
    return [turn for turn in history if isinstance(turn, ConversationTurn)]
