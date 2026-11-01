"""Conversation-thread endpoints backed by the workflow checkpointer."""

from typing import Annotated, Any

from fastapi import APIRouter, Path, Response, status

from insightflow.agents.checkpoint import read_thread_history
from insightflow.api.dependencies import CheckpointerDependency
from insightflow.api.schemas import ConversationTurnResponse, ErrorResponse, ThreadResponse
from insightflow.core.exceptions import ThreadNotFoundError

router = APIRouter(tags=["threads"])

ThreadIdPath = Annotated[
    str, Path(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
]
NOT_FOUND: dict[int | str, dict[str, Any]] = {
    404: {
        "model": ErrorResponse,
        "description": "The thread does not exist, or conversation storage is disabled.",
    }
}


@router.get("/threads/{thread_id}", response_model=ThreadResponse, responses=NOT_FOUND)
async def get_thread(
    thread_id: ThreadIdPath, checkpointer: CheckpointerDependency
) -> ThreadResponse:
    """Return the completed turns of a conversation thread."""
    if checkpointer is None:
        raise ThreadNotFoundError
    history = await read_thread_history(checkpointer, thread_id)
    if history is None:
        raise ThreadNotFoundError
    return ThreadResponse(
        thread_id=thread_id,
        turns=[ConversationTurnResponse(query=turn.query, answer=turn.answer) for turn in history],
    )


@router.delete(
    "/threads/{thread_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    responses=NOT_FOUND,
)
async def delete_thread(thread_id: ThreadIdPath, checkpointer: CheckpointerDependency) -> None:
    """Delete every stored checkpoint of a conversation thread."""
    if checkpointer is None or await read_thread_history(checkpointer, thread_id) is None:
        raise ThreadNotFoundError
    await checkpointer.adelete_thread(thread_id)
