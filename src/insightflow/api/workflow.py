"""Compose the agent workflow from settings and adapters, and map its state to the API."""

import uuid

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver

from insightflow.agents.checkpoint import thread_config
from insightflow.agents.document_search import DocumentSearch
from insightflow.agents.graph import AgentGraph, WorkflowNodes, build_graph, recursion_limit
from insightflow.agents.models import SEARCH_TOOL
from insightflow.agents.nodes.act import make_act_node
from insightflow.agents.nodes.agent import make_agent_node
from insightflow.agents.nodes.answer import make_answer_node
from insightflow.agents.nodes.finalize import finalize_node
from insightflow.agents.nodes.retry import retry_node
from insightflow.agents.nodes.verify import make_verify_node
from insightflow.agents.observations import (
    NumberedSource,
    number_sources,
    observation_keys,
    source_key,
)
from insightflow.agents.state import AgentState
from insightflow.api.schemas import (
    ExecutionMetadata,
    QueryResponse,
    Source,
    SourceReference,
    StepReport,
    VerificationReport,
    WebSourceReference,
)
from insightflow.core.config import Settings
from insightflow.providers.llm import ChatProvider
from insightflow.rag.models import ContextBudget
from insightflow.rag.protocols import ContextAssembler, Retriever
from insightflow.tools.registry import ToolSet


def build_workflow(
    settings: Settings,
    chat_provider: ChatProvider,
    retriever: Retriever,
    assembler: ContextAssembler,
    checkpointer: BaseCheckpointSaver[str] | None,
    *,
    tools: ToolSet | None = None,
    stream_tokens: bool = False,
) -> AgentGraph:
    """Build the ReAct workflow for one request."""
    tool_set = tools if tools is not None else ToolSet()
    search = DocumentSearch(
        retriever,
        assembler,
        top_k=settings.rag_candidate_top_k,
        budget=ContextBudget(
            max_tokens=settings.rag_context_token_budget,
            max_chunks=settings.rag_generation_max_chunks,
        ),
    )
    nodes = WorkflowNodes(
        agent=make_agent_node(
            chat_provider,
            tool_set,
            max_steps=settings.agent_max_steps,
            history_turns=settings.agent_history_turns,
        ),
        act=make_act_node(search, tool_set),
        answer=make_answer_node(
            chat_provider,
            history_turns=settings.agent_history_turns,
            stream_tokens=stream_tokens,
        ),
        verify=make_verify_node(
            chat_provider if settings.agent_verifier_llm_enabled else None
        ),
        retry=retry_node,
        finalize=finalize_node,
    )
    return build_graph(nodes, max_attempts=settings.agent_max_attempts, checkpointer=checkpointer)


def resolve_thread(
    requested: str | None,
    checkpointer: BaseCheckpointSaver[str] | None,
    settings: Settings,
) -> tuple[str | None, RunnableConfig]:
    """Return the thread to use (`None` when stateless) and the run configuration."""
    limit = recursion_limit(settings.agent_max_steps, settings.agent_max_attempts)
    if checkpointer is None:
        return None, {"recursion_limit": limit}
    thread_id = requested or uuid.uuid4().hex
    return thread_id, {**thread_config(thread_id), "recursion_limit": limit}


def to_query_response(
    state: AgentState,
    *,
    thread_id: str | None,
    duration_ms: int,
) -> QueryResponse:
    """Map the final workflow state to the public response."""
    observations = state.get("observations", [])
    numbered = number_sources(observations)
    number_by_key = {source_key(source): source.number for source in numbered}
    steps = [
        StepReport(
            attempt=observation.attempt,
            thought=observation.thought,
            tool=observation.result.tool,
            server=observation.result.server,
            arguments=observation.result.arguments,
            call_id=observation.result.call_id,
            status=observation.result.status,
            error=observation.result.error,
            truncated=observation.result.truncated,
            sources=sorted({number_by_key[key] for key in observation_keys(observation)}),
            duration_ms=observation.result.duration_ms,
        )
        for observation in observations
    ]
    verification = state.get("verification")
    return QueryResponse(
        answer=state.get("answer", ""),
        sources=[_source_reference(source) for source in numbered],
        thread_id=thread_id,
        verification=VerificationReport(
            status=verification.status if verification else "skipped",
            method=verification.method if verification else "none",
            issues=list(verification.issues) if verification else [],
        ),
        metadata=ExecutionMetadata(
            attempts=state.get("attempt", 1),
            retrieval_queries=[
                str(step.arguments.get("query", ""))
                for step in steps
                if step.tool == SEARCH_TOOL and step.status == "ok"
            ],
            steps=steps,
            nodes=state.get("node_trace", []),
            duration_ms=duration_ms,
        ),
    )


def _source_reference(source: NumberedSource) -> Source:
    if source.passage is not None:
        item = source.passage
        return SourceReference(
            citation=source.number,
            source_identifier=item.source_identifier,
            source_name=item.source_name,
            chunk_id=item.chunk.chunk_id,
            page_start=item.chunk.page_start,
            page_end=item.chunk.page_end,
            score=item.score,
        )
    assert source.page is not None
    return WebSourceReference(citation=source.number, title=source.page.title, url=source.page.url)
