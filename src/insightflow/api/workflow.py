"""Compose the agent workflow from settings and adapters, and map its state to the API."""

import uuid

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver

from insightflow.agents.checkpoint import thread_config
from insightflow.agents.graph import AgentGraph, WorkflowNodes, build_graph
from insightflow.agents.nodes.answer import make_answer_node
from insightflow.agents.nodes.finalize import finalize_node
from insightflow.agents.nodes.retrieve import make_retrieve_node
from insightflow.agents.nodes.rewrite import make_rewrite_node
from insightflow.agents.nodes.route import make_route_node
from insightflow.agents.nodes.verify import make_verify_node
from insightflow.agents.state import AgentState
from insightflow.api.schemas import (
    ExecutionMetadata,
    QueryResponse,
    SourceReference,
    VerificationReport,
)
from insightflow.core.config import Settings
from insightflow.providers.llm import ChatProvider
from insightflow.rag.models import ContextBudget
from insightflow.rag.protocols import ContextAssembler, Retriever


def build_workflow(
    settings: Settings,
    chat_provider: ChatProvider,
    retriever: Retriever,
    assembler: ContextAssembler,
    checkpointer: BaseCheckpointSaver[str] | None,
    *,
    stream_tokens: bool = False,
) -> AgentGraph:
    """Build the routed, verified workflow for one request."""
    history_turns = settings.agent_history_turns
    nodes = WorkflowNodes(
        route=make_route_node(
            chat_provider if settings.agent_router_enabled else None,
            history_turns=history_turns,
        ),
        retrieve=make_retrieve_node(
            retriever,
            assembler,
            top_k=settings.rag_candidate_top_k,
            budget=ContextBudget(
                max_tokens=settings.rag_context_token_budget,
                max_chunks=settings.rag_generation_max_chunks,
            ),
        ),
        answer=make_answer_node(
            chat_provider, history_turns=history_turns, stream_tokens=stream_tokens
        ),
        verify=make_verify_node(
            chat_provider if settings.agent_verifier_llm_enabled else None
        ),
        rewrite=make_rewrite_node(chat_provider),
        finalize=finalize_node,
    )
    return build_graph(
        nodes, max_attempts=settings.agent_max_attempts, checkpointer=checkpointer
    )


def resolve_thread(
    requested: str | None,
    checkpointer: BaseCheckpointSaver[str] | None,
) -> tuple[str | None, RunnableConfig | None]:
    """Return the thread to use and its run configuration; both are `None` when stateless."""
    if checkpointer is None:
        return None, None
    thread_id = requested or uuid.uuid4().hex
    return thread_id, thread_config(thread_id)


def to_query_response(
    state: AgentState,
    *,
    thread_id: str | None,
    duration_ms: int,
) -> QueryResponse:
    """Map the final workflow state to the public response."""
    context = state.get("context")
    sources = [
        SourceReference(
            citation=citation,
            source_identifier=item.source_identifier,
            source_name=item.source_name,
            chunk_id=item.chunk.chunk_id,
            page_start=item.chunk.page_start,
            page_end=item.chunk.page_end,
            score=item.score,
        )
        for citation, item in enumerate(context.chunks if context else [], start=1)
    ]
    verification = state.get("verification")
    return QueryResponse(
        answer=state.get("answer", ""),
        sources=sources,
        thread_id=thread_id,
        verification=VerificationReport(
            status=verification.status if verification else "skipped",
            method=verification.method if verification else "none",
            issues=list(verification.issues) if verification else [],
        ),
        metadata=ExecutionMetadata(
            route=state.get("route") or "retrieve",
            attempts=state.get("attempt", 1),
            retrieval_queries=state.get("retrieval_queries", []),
            nodes=state.get("node_trace", []),
            duration_ms=duration_ms,
        ),
    )
