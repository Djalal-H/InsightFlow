"""Query endpoints: the routed, verified retrieval-augmented answer workflow."""

import time
from collections.abc import AsyncIterator
from typing import Any, cast

from fastapi import APIRouter
from fastapi.sse import EventSourceResponse, ServerSentEvent

from insightflow.agents.state import AgentState, new_turn_input
from insightflow.api.dependencies import (
    ChatProviderDependency,
    CheckpointerDependency,
    ContextAssemblerDependency,
    RetrieverDependency,
    SettingsDependency,
    ToolSetDependency,
)
from insightflow.api.errors import describe_error
from insightflow.api.schemas import ErrorResponse, QueryRequest, QueryResponse
from insightflow.api.workflow import build_workflow, resolve_thread, to_query_response
from insightflow.core.exceptions import InsightFlowError

router = APIRouter(tags=["query"])

ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    429: {"model": ErrorResponse, "description": "The provider is rate limited."},
    502: {"model": ErrorResponse, "description": "The provider request failed."},
    503: {
        "model": ErrorResponse,
        "description": "The provider is not configured or the document store is unavailable.",
    },
    504: {"model": ErrorResponse, "description": "The provider request timed out."},
}


def elapsed_ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


@router.post("/query", response_model=QueryResponse, responses=ERROR_RESPONSES)
async def query(
    request: QueryRequest,
    chat_provider: ChatProviderDependency,
    retriever: RetrieverDependency,
    assembler: ContextAssemblerDependency,
    settings: SettingsDependency,
    checkpointer: CheckpointerDependency,
    tools: ToolSetDependency,
) -> QueryResponse:
    """Let the agent search documents and call tools, then answer, verify, and cite sources."""
    started = time.perf_counter()
    thread_id, config = resolve_thread(request.thread_id, checkpointer, settings)
    graph = build_workflow(
        settings, chat_provider, retriever, assembler, checkpointer, tools=tools
    )

    result = cast(AgentState, await graph.ainvoke(new_turn_input(request.query), config))

    return to_query_response(result, thread_id=thread_id, duration_ms=elapsed_ms(started))


def node_event(node: str, update: AgentState) -> dict[str, Any]:
    """Summarize one node update for a `node` event without sending internal state."""
    data: dict[str, Any] = {"node": node}
    if node == "agent" and update.get("pending_action") is not None:
        action = update["pending_action"]
        data["action"] = action.action if action is not None else None
    elif node == "act" and update.get("observations"):
        observation = update["observations"][-1]
        data["tool"] = observation.result.tool
        data["status"] = observation.result.status
    elif node == "verify" and update.get("verification") is not None:
        verification = update["verification"]
        data["status"] = verification.status if verification is not None else None
    elif node == "retry" and "attempt" in update:
        data["attempt"] = update["attempt"]
    return data


@router.post(
    "/query/stream",
    response_class=EventSourceResponse,
    responses={
        **ERROR_RESPONSES,
        200: {
            "description": (
                "Server-Sent Events: `node` after each workflow step, `attempt` when a retry "
                "starts (discard earlier tokens), `token` for answer text, then one `final` "
                "event with the QueryResponse, or one `error` event with an ErrorResponse."
            ),
        },
    },
)
async def query_stream(
    request: QueryRequest,
    chat_provider: ChatProviderDependency,
    retriever: RetrieverDependency,
    assembler: ContextAssemblerDependency,
    settings: SettingsDependency,
    checkpointer: CheckpointerDependency,
    tools: ToolSetDependency,
) -> AsyncIterator[ServerSentEvent]:
    """Run the same workflow as POST /query and stream its progress and answer tokens."""
    started = time.perf_counter()
    thread_id, config = resolve_thread(request.thread_id, checkpointer, settings)
    graph = build_workflow(
        settings,
        chat_provider,
        retriever,
        assembler,
        checkpointer,
        tools=tools,
        stream_tokens=True,
    )

    final_state: AgentState = {}
    try:
        async for mode, chunk in graph.astream(
            new_turn_input(request.query),
            config,
            stream_mode=["updates", "custom", "values"],
        ):
            if mode == "custom":
                event = cast(dict[str, Any], chunk)
                yield ServerSentEvent(
                    event="token", data={"attempt": event["attempt"], "text": event["text"]}
                )
            elif mode == "updates":
                for node, update in cast(dict[str, AgentState], chunk).items():
                    yield ServerSentEvent(event="node", data=node_event(node, update))
                    if node == "retry":
                        yield ServerSentEvent(event="attempt", data={"attempt": update["attempt"]})
            elif mode == "values":
                final_state = cast(AgentState, chunk)
    except InsightFlowError as exc:
        _, detail = describe_error(exc)
        yield ServerSentEvent(event="error", data={"error": detail})
        return

    response = to_query_response(final_state, thread_id=thread_id, duration_ms=elapsed_ms(started))
    yield ServerSentEvent(event="final", data=response.model_dump(mode="json"))
