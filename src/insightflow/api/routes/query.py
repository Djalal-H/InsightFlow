"""Query endpoint: the primary agent answer flow."""

from fastapi import APIRouter

from insightflow.agents.graph import build_graph
from insightflow.agents.nodes.answer import make_answer_node
from insightflow.agents.state import AgentState
from insightflow.api.dependencies import ChatProviderDependency
from insightflow.api.schemas import ErrorResponse, QueryRequest, QueryResponse

router = APIRouter(tags=["query"])


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        429: {"model": ErrorResponse, "description": "The provider is rate limited."},
        502: {"model": ErrorResponse, "description": "The provider request failed."},
        503: {"model": ErrorResponse, "description": "The provider is not configured."},
        504: {"model": ErrorResponse, "description": "The provider request timed out."},
    },
)
async def query(request: QueryRequest, chat_provider: ChatProviderDependency) -> QueryResponse:
    """Answer a user query through the LangGraph agent workflow."""
    graph = build_graph(make_answer_node(chat_provider))

    initial_state: AgentState = {"query": request.query}
    result = await graph.ainvoke(initial_state)

    return QueryResponse(
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
    )
