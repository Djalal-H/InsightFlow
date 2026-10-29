"""Query endpoint: the grounded retrieval-augmented answer flow."""

from fastapi import APIRouter

from insightflow.agents.graph import build_graph
from insightflow.agents.nodes.answer import make_answer_node
from insightflow.agents.nodes.retrieve import make_retrieve_node
from insightflow.agents.state import AgentState
from insightflow.api.dependencies import (
    ChatProviderDependency,
    ContextAssemblerDependency,
    RetrieverDependency,
    SettingsDependency,
)
from insightflow.api.schemas import ErrorResponse, QueryRequest, QueryResponse, SourceReference
from insightflow.rag.models import ContextBudget

router = APIRouter(tags=["query"])


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        429: {"model": ErrorResponse, "description": "The provider is rate limited."},
        502: {"model": ErrorResponse, "description": "The provider request failed."},
        503: {
            "model": ErrorResponse,
            "description": "The provider is not configured or the document store is unavailable.",
        },
        504: {"model": ErrorResponse, "description": "The provider request timed out."},
    },
)
async def query(
    request: QueryRequest,
    chat_provider: ChatProviderDependency,
    retriever: RetrieverDependency,
    assembler: ContextAssemblerDependency,
    settings: SettingsDependency,
) -> QueryResponse:
    """Retrieve indexed context, answer from it, and return the cited sources."""
    retrieve_node = make_retrieve_node(
        retriever,
        assembler,
        top_k=settings.rag_candidate_top_k,
        budget=ContextBudget(
            max_tokens=settings.rag_context_token_budget,
            max_chunks=settings.rag_generation_max_chunks,
        ),
    )
    graph = build_graph(retrieve_node, make_answer_node(chat_provider))

    initial_state: AgentState = {"query": request.query}
    result = await graph.ainvoke(initial_state)

    context = result.get("context")
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
    return QueryResponse(answer=result.get("answer", ""), sources=sources)
