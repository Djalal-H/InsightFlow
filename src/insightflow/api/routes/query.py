"""Query endpoint: the primary agent answer flow."""

from fastapi import APIRouter

from insightflow.agents.graph import build_graph
from insightflow.agents.nodes.answer import make_answer_node
from insightflow.agents.state import AgentState
from insightflow.api.schemas import QueryRequest, QueryResponse
from insightflow.core.config import get_settings
from insightflow.providers.llm import LiteLLMChatProvider

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Answer a user query through the LangGraph agent workflow."""
    settings = get_settings()
    chat_provider = LiteLLMChatProvider(
        model=settings.litellm_chat_model,
        api_key=settings.litellm_api_key,
        api_base=settings.litellm_api_base,
    )
    graph = build_graph(make_answer_node(chat_provider))

    initial_state: AgentState = {"query": request.query}
    result = await graph.ainvoke(initial_state)

    return QueryResponse(
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
    )
