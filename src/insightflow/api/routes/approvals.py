"""Decide a paused consequential tool call and continue the run."""

import time
from typing import Annotated, cast

from fastapi import APIRouter, Path
from langgraph.types import Command

from insightflow.agents.checkpoint import read_thread
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
from insightflow.api.routes.query import ERROR_RESPONSES, elapsed_ms, request_recorder
from insightflow.api.routes.threads import ThreadIdPath
from insightflow.api.schemas import ApprovalDecision, QueryRequest, QueryResponse
from insightflow.api.security import AdmittedPrincipal
from insightflow.api.workflow import (
    build_workflow,
    charge_usage,
    finish_run,
    require_thread_access,
    resolve_thread,
    start_run,
    to_query_response,
)
from insightflow.core.exceptions import (
    ApprovalNotFoundError,
    AuthorizationError,
    InsightFlowError,
    ThreadNotFoundError,
)
from insightflow.security.audit import audit
from insightflow.security.auth import TOOLS_CONSEQUENTIAL

router = APIRouter(tags=["threads"])

ApprovalIdPath = Annotated[str, Path(pattern=r"^[0-9a-f]{32}$")]


@router.post(
    "/threads/{thread_id}/approvals/{approval_id}",
    response_model=QueryResponse,
    responses=ERROR_RESPONSES,
)
async def decide_approval(
    thread_id: ThreadIdPath,
    approval_id: ApprovalIdPath,
    decision: ApprovalDecision,
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
    """Approve or reject the pending call, then continue the run to its answer.

    Approving needs the `tools:consequential` scope; rejecting does not. The decision counts
    only for this approval ID. A new POST /query on the thread drops the pending approval.
    """
    started = time.perf_counter()
    recorder = request_recorder()
    thread = await read_thread(checkpointer, thread_id) if checkpointer is not None else None
    if checkpointer is None or thread is None:
        raise ThreadNotFoundError
    require_thread_access(principal, thread.owner)
    approve = decision.decision == "approve"
    if approve and not principal.has(TOOLS_CONSEQUENTIAL):
        audit("access_denied", missing_scope=TOOLS_CONSEQUENTIAL)
        raise AuthorizationError(TOOLS_CONSEQUENTIAL)

    _, config = resolve_thread(thread_id, checkpointer, settings)
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
    snapshot = await graph.aget_state(config)
    waiting = any(
        isinstance(item.value, dict) and item.value.get("approval_id") == approval_id
        for item in snapshot.interrupts
    )
    if not waiting:
        raise ApprovalNotFoundError
    # The tool set may not have listed its servers since a restart; the call needs the owner.
    await tools.list_tools()

    start_run(recorder, thread_id, sink)
    request = QueryRequest(query=str(snapshot.values.get("query", "")) or "-", thread_id=thread_id)
    try:
        result = cast(
            AgentState,
            await graph.ainvoke(
                Command(resume={"approval_id": approval_id, "approved": approve}), config
            ),
        )
    except InsightFlowError:
        finish_run(recorder, sink, request, None, name="approval")
        raise
    finally:
        await charge_usage(limiter, principal, recorder)

    response = to_query_response(
        result, thread_id=thread_id, duration_ms=elapsed_ms(started), recorder=recorder
    )
    finish_run(recorder, sink, request, response, name="approval")
    return response
