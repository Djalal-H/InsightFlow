"""Query endpoints: the routed, verified retrieval-augmented answer workflow."""

import time
from collections.abc import AsyncIterator
from typing import Any, cast

from fastapi import APIRouter
from fastapi.sse import EventSourceResponse, ServerSentEvent

from insightflow.agents.state import AgentState
from insightflow.api.dependencies import (
    ChatProviderDependency,
    CheckpointerDependency,
    ContextAssemblerDependency,
    ConversationMemoryDependency,
    RateLimiterDependency,
    RetrieverDependency,
    SafetyClassifierDependency,
    SettingsDependency,
    ToolSetDependency,
    TraceSinkDependency,
)
from insightflow.api.errors import report_error
from insightflow.api.schemas import ErrorResponse, QueryRequest, QueryResponse
from insightflow.api.security import AdmittedPrincipal
from insightflow.api.workflow import (
    build_workflow,
    charge_usage,
    finish_run,
    open_thread,
    screen_query,
    start_run,
    to_query_response,
    turn_input,
)
from insightflow.core.exceptions import InsightFlowError
from insightflow.core.observability import RunRecorder, current_recorder

router = APIRouter(tags=["query"])

ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorResponse, "description": "A content policy refused the question."},
    401: {"model": ErrorResponse, "description": "The bearer token is missing or not valid."},
    403: {"model": ErrorResponse, "description": "The token does not have the `query` scope."},
    404: {"model": ErrorResponse, "description": "The thread belongs to another user."},
    413: {"model": ErrorResponse, "description": "The request body is too large."},
    429: {
        "model": ErrorResponse,
        "description": (
            "The user's rate or daily quota is used up (`rate_limited`, `quota_exceeded`, with "
            "Retry-After), or the provider is rate limited."
        ),
    },
    502: {"model": ErrorResponse, "description": "The provider request failed."},
    503: {
        "model": ErrorResponse,
        "description": (
            "The provider is not configured, or the document store or rate limiter is "
            "unavailable."
        ),
    },
    504: {"model": ErrorResponse, "description": "The provider request timed out."},
}


def elapsed_ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


def request_recorder() -> RunRecorder:
    """The run opened by `RequestContextMiddleware`; a detached one if the app has none."""
    return current_recorder() or RunRecorder()


@router.post("/query", response_model=QueryResponse, responses=ERROR_RESPONSES)
async def query(
    request: QueryRequest,
    principal: AdmittedPrincipal,
    chat_provider: ChatProviderDependency,
    retriever: RetrieverDependency,
    assembler: ContextAssemblerDependency,
    settings: SettingsDependency,
    checkpointer: CheckpointerDependency,
    tools: ToolSetDependency,
    memory: ConversationMemoryDependency,
    sink: TraceSinkDependency,
    limiter: RateLimiterDependency,
    safety: SafetyClassifierDependency,
) -> QueryResponse:
    """Let the agent search documents and call tools, then answer, verify, and cite sources.

    A consequential tool call pauses the run: the response then has `status`
    `approval_required` and the call to approve.
    """
    started = time.perf_counter()
    recorder = request_recorder()
    access = await open_thread(request.thread_id, checkpointer, settings, principal)
    start_run(recorder, access.thread_id, sink)
    request, signals = await screen_query(request, settings, safety)
    graph = build_workflow(
        settings,
        chat_provider,
        retriever,
        assembler,
        checkpointer,
        tools=tools,
        memory=memory,
        safety=safety,
    )
    graph_input = turn_input(
        request, access.thread_id, principal=principal, owner=access.owner, guard_signals=signals
    )

    try:
        result = cast(AgentState, await graph.ainvoke(graph_input, access.config))
    except InsightFlowError:
        finish_run(recorder, sink, request, None, name="query")
        raise
    finally:
        await charge_usage(limiter, principal, recorder)

    response = to_query_response(
        result, thread_id=access.thread_id, duration_ms=elapsed_ms(started), recorder=recorder
    )
    finish_run(recorder, sink, request, response, name="query")
    return response


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
    elif node == "recall":
        data["recalled_turns"] = len(update.get("recalled", []))
    elif node == "retry" and "attempt" in update:
        data["attempt"] = update["attempt"]
    elif node == "approve":
        data["status"] = "approved" if update.get("approved_call") else "refused"
    return data


@router.post(
    "/query/stream",
    response_class=EventSourceResponse,
    responses={
        **ERROR_RESPONSES,
        200: {
            "description": (
                "Server-Sent Events: `node` after each workflow step, `attempt` when a retry "
                "starts (discard earlier tokens), `token` for answer text, `answer_replaced` "
                "when a guardrail changed the answer (discard this attempt's tokens), "
                "`approval_required` when a consequential tool call waits for a decision, then "
                "one `final` event with the QueryResponse (its `answer` is authoritative), or "
                "one `error` event with an ErrorResponse."
            ),
        },
    },
)
async def query_stream(
    request: QueryRequest,
    principal: AdmittedPrincipal,
    chat_provider: ChatProviderDependency,
    retriever: RetrieverDependency,
    assembler: ContextAssemblerDependency,
    settings: SettingsDependency,
    checkpointer: CheckpointerDependency,
    tools: ToolSetDependency,
    memory: ConversationMemoryDependency,
    sink: TraceSinkDependency,
    limiter: RateLimiterDependency,
    safety: SafetyClassifierDependency,
) -> AsyncIterator[ServerSentEvent]:
    """Run the same workflow as POST /query and stream its progress and answer tokens."""
    started = time.perf_counter()
    recorder = request_recorder()
    try:
        # The response has started: failures before the run are `error` events too.
        access = await open_thread(request.thread_id, checkpointer, settings, principal)
        start_run(recorder, access.thread_id, sink)
        request, signals = await screen_query(request, settings, safety)
    except InsightFlowError as exc:
        _, detail = report_error(exc)
        yield ServerSentEvent(event="error", data={"error": detail})
        return
    graph = build_workflow(
        settings,
        chat_provider,
        retriever,
        assembler,
        checkpointer,
        tools=tools,
        memory=memory,
        stream_tokens=True,
        safety=safety,
    )
    graph_input = turn_input(
        request, access.thread_id, principal=principal, owner=access.owner, guard_signals=signals
    )

    final_state: AgentState = {}
    try:
        async for mode, chunk in graph.astream(
            graph_input,
            access.config,
            stream_mode=["updates", "custom", "values"],
        ):
            if mode == "custom":
                event = cast(dict[str, Any], chunk)
                if event.get("type") == "replace":
                    yield ServerSentEvent(
                        event="answer_replaced", data={"attempt": event["attempt"]}
                    )
                else:
                    yield ServerSentEvent(
                        event="token", data={"attempt": event["attempt"], "text": event["text"]}
                    )
            elif mode == "updates":
                for node, update in cast(dict[str, AgentState], chunk).items():
                    if node == "__interrupt__":
                        continue
                    yield ServerSentEvent(event="node", data=node_event(node, update))
                    if node == "retry":
                        yield ServerSentEvent(event="attempt", data={"attempt": update["attempt"]})
            elif mode == "values":
                final_state = cast(AgentState, chunk)
    except InsightFlowError as exc:
        _, detail = report_error(exc)
        finish_run(recorder, sink, request, None, name="query_stream")
        yield ServerSentEvent(event="error", data={"error": detail})
        return
    finally:
        await charge_usage(limiter, principal, recorder)

    response = to_query_response(
        final_state, thread_id=access.thread_id, duration_ms=elapsed_ms(started), recorder=recorder
    )
    finish_run(recorder, sink, request, response, name="query_stream")
    if response.pending_approval is not None:
        yield ServerSentEvent(
            event="approval_required", data=response.pending_approval.model_dump(mode="json")
        )
    yield ServerSentEvent(event="final", data=response.model_dump(mode="json"))
