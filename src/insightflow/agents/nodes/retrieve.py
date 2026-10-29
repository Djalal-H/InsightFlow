"""Retrieve node: embeds the query, searches stored chunks, and assembles context."""

from insightflow.agents.graph import AgentNode
from insightflow.agents.state import AgentState
from insightflow.rag.models import ContextBudget, RetrievalQuery
from insightflow.rag.protocols import ContextAssembler, Retriever


def make_retrieve_node(
    retriever: Retriever,
    assembler: ContextAssembler,
    *,
    top_k: int,
    budget: ContextBudget,
) -> AgentNode:
    """Bind retrieval components and return a LangGraph-compatible retrieve node."""

    async def retrieve_node(state: AgentState) -> AgentState:
        query = state["query"]
        candidates = await retriever.retrieve(RetrievalQuery(text=query, top_k=top_k))
        return {"context": assembler.assemble(query, candidates, budget)}

    return retrieve_node
