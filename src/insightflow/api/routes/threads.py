"""Conversation-thread endpoints backed by the workflow checkpointer."""

from typing import Annotated, Any

from fastapi import APIRouter, Path, Response, status

from insightflow.agents.checkpoint import read_thread
from insightflow.api.dependencies import CheckpointerDependency, ConversationMemoryDependency
from insightflow.api.schemas import ConversationTurnResponse, ErrorResponse, ThreadResponse
from insightflow.api.security import ThreadDeleter, ThreadReader
from insightflow.api.workflow import require_thread_access
from insightflow.core.exceptions import ThreadNotFoundError

router = APIRouter(tags=["threads"])

ThreadIdPath = Annotated[
    str, Path(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
]
NOT_FOUND: dict[int | str, dict[str, Any]] = {
    401: {"model": ErrorResponse, "description": "The bearer token is missing or not valid."},
    403: {"model": ErrorResponse, "description": "The token does not have the scope."},
    404: {
        "model": ErrorResponse,
        "description": (
            "The thread does not exist, belongs to another user, or conversation storage is "
            "disabled."
        ),
    },
}


@router.get("/threads/{thread_id}", response_model=ThreadResponse, responses=NOT_FOUND)
async def get_thread(
    thread_id: ThreadIdPath, checkpointer: CheckpointerDependency, principal: ThreadReader
) -> ThreadResponse:
    """Return the completed turns of a conversation thread of the caller."""
    if checkpointer is None:
        raise ThreadNotFoundError
    thread = await read_thread(checkpointer, thread_id)
    if thread is None:
        raise ThreadNotFoundError
    require_thread_access(principal, thread.owner)
    return ThreadResponse(
        thread_id=thread_id,
        turns=[
            ConversationTurnResponse(query=turn.query, answer=turn.answer)
            for turn in thread.history
        ],
    )


@router.delete(
    "/threads/{thread_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    responses=NOT_FOUND,
)
async def delete_thread(
    thread_id: ThreadIdPath,
    checkpointer: CheckpointerDependency,
    memory: ConversationMemoryDependency,
    principal: ThreadDeleter,
) -> None:
    """Delete every stored checkpoint and remembered turn of a conversation thread."""
    thread = await read_thread(checkpointer, thread_id) if checkpointer is not None else None
    if checkpointer is None or thread is None:
        raise ThreadNotFoundError
    require_thread_access(principal, thread.owner)
    if memory is not None:
        # Memory first: if it fails, the thread still exists and the delete can be repeated.
        await memory.forget(thread_id)
    await checkpointer.adelete_thread(thread_id)
