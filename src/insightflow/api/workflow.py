"""Compose the agent workflow from settings and adapters, and map its state to the API."""

import logging
import uuid

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver

from insightflow.agents.checkpoint import thread_config
from insightflow.agents.document_search import DocumentSearch
from insightflow.agents.graph import AgentGraph, WorkflowNodes, build_graph, recursion_limit
from insightflow.agents.memory import ConversationMemory
from insightflow.agents.models import SEARCH_TOOL
from insightflow.agents.nodes.act import make_act_node
from insightflow.agents.nodes.agent import make_agent_node
from insightflow.agents.nodes.answer import make_answer_node
from insightflow.agents.nodes.finalize import make_finalize_node
from insightflow.agents.nodes.recall import make_recall_node
from insightflow.agents.nodes.retry import retry_node
from insightflow.agents.nodes.verify import make_verify_node
from insightflow.agents.observations import (
    NumberedSource,
    number_sources,
    observation_keys,
    source_key,
)
from insightflow.agents.state import AgentState, new_turn_input
from insightflow.api.schemas import (
    ExecutionMetadata,
    ProviderUsage,
    QueryRequest,
    QueryResponse,
    Source,
    SourceReference,
    StepReport,
    UsageReport,
    VerificationReport,
    WebSourceReference,
)
from insightflow.core.config import Settings
from insightflow.core.observability import (
    PROVIDER_KINDS,
    RunRecorder,
    TraceSink,
    export_run,
)
from insightflow.providers.instrumented import instrument_chat
from insightflow.providers.llm import ChatProvider
from insightflow.rag.models import ContextBudget
from insightflow.rag.protocols import ContextAssembler, Retriever
from insightflow.tools.registry import ToolSet

logger = logging.getLogger("insightflow.query")


def build_workflow(
    settings: Settings,
    chat_provider: ChatProvider,
    retriever: Retriever,
    assembler: ContextAssembler,
    checkpointer: BaseCheckpointSaver[str] | None,
    *,
    tools: ToolSet | None = None,
    memory: ConversationMemory | None = None,
    stream_tokens: bool = False,
) -> AgentGraph:
    """Build the ReAct workflow for one request.

    Conversation memory needs a thread, so `memory` is ignored without a checkpointer.
    """
    if checkpointer is None:
        memory = None
    chat_provider = instrument_chat(chat_provider, model=settings.litellm_chat_model)
    tool_set = tools if tools is not None else ToolSet()
    search = DocumentSearch(
        retriever,
        assembler,
        top_k=settings.rag_candidate_top_k,
        budget=ContextBudget(
            max_tokens=settings.rag_context_token_budget,
            max_chunks=settings.rag_generation_max_chunks,
        ),
        filters_enabled=settings.rag_search_filters_enabled,
    )
    nodes = WorkflowNodes(
        agent=make_agent_node(
            chat_provider,
            tool_set,
            max_steps=settings.agent_max_steps,
            history_turns=settings.agent_history_turns,
            search_spec=search.spec,
            multi_hop=settings.agent_multi_hop_enabled,
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
        finalize=make_finalize_node(memory),
        recall=(
            make_recall_node(memory, history_turns=settings.agent_history_turns)
            if memory is not None
            else None
        ),
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


def turn_input(request: QueryRequest, thread_id: str | None) -> AgentState:
    """Return the graph input for a request, with its document filters as the search scope."""
    scope: dict[str, list[str]] = {}
    if request.filters is not None:
        if request.filters.file_types:
            scope["file_types"] = list(dict.fromkeys(request.filters.file_types))
        if request.filters.sources:
            scope["sources"] = list(dict.fromkeys(request.filters.sources))
    return new_turn_input(request.query, thread_id=thread_id, search_scope=scope)


def to_query_response(
    state: AgentState,
    *,
    thread_id: str | None,
    duration_ms: int,
    recorder: RunRecorder | None = None,
) -> QueryResponse:
    """Map the final workflow state, and the run's records when given, to the public response."""
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
            recalled_turns=len(state.get("recalled", [])),
            duration_ms=duration_ms,
            usage=usage_report(state, recorder),
        ),
    )


def usage_report(state: AgentState, recorder: RunRecorder | None) -> UsageReport:
    """Counters from the final state, and provider and store totals from the run's records."""
    observations = state.get("observations", [])
    searches = [item for item in observations if item.result.tool == SEARCH_TOOL]
    report = UsageReport(
        retrieval_calls=len(searches),
        retrieved_passages=sum(len(item.passages) for item in searches),
        tool_calls=len(observations) - len(searches),
        tool_errors=sum(1 for item in observations if item.result.status == "error"),
        retries=max(state.get("attempt", 1) - 1, 0),
    )
    if recorder is None:
        return report
    providers: dict[str, ProviderUsage] = {}
    for kind in PROVIDER_KINDS:
        totals = recorder.totals(kind)
        providers[kind] = ProviderUsage(
            calls=totals.calls,
            errors=totals.errors,
            prompt_tokens=totals.prompt_tokens,
            completion_tokens=totals.completion_tokens,
            unreported_usage=totals.unreported_usage,
        )
    return report.model_copy(
        update={
            **providers,
            "store_calls": len(recorder.store_calls),
            "errors": dict(recorder.errors),
            "node_timings_ms": recorder.node_timings_ms(),
        }
    )


def start_run(recorder: RunRecorder, thread_id: str | None, sink: TraceSink) -> None:
    """Attach the request's thread to its run, and capture content only when the sink wants it."""
    recorder.thread_id = thread_id
    recorder.capture_content = sink.capture_content


def finish_run(
    recorder: RunRecorder,
    sink: TraceSink,
    request: QueryRequest,
    response: QueryResponse | None,
    *,
    name: str,
) -> None:
    """Log the request's summary line and send the run to the trace sink."""
    fields = {**recorder.summary(), "outcome": "ok" if response is not None else "error"}
    if response is not None:
        fields.update(
            attempts=response.metadata.attempts,
            steps=len(response.metadata.steps),
            sources=len(response.sources),
            verification=response.verification.status,
            duration_ms=response.metadata.duration_ms,
        )
    logger.info("query completed", extra={"fields": fields})
    export_run(
        sink,
        recorder,
        name=name,
        input=request.query,
        output=response.answer if response is not None else None,
        metadata={
            "attempts": response.metadata.attempts if response is not None else None,
            "verification": response.verification.status if response is not None else None,
        },
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
