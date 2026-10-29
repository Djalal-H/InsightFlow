"""LangGraph construction helpers."""

from collections.abc import Awaitable, Callable

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from insightflow.agents.state import AgentState

AgentNode = Callable[[AgentState], Awaitable[AgentState]]


def build_graph(
    retrieve_node: AgentNode,
    answer_node: AgentNode,
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    """Build the grounded-answer workflow: retrieve context, then answer from it."""
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)  # type: ignore[call-overload]
    graph.add_node("answer", answer_node)  # type: ignore[call-overload]
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)
    return graph.compile()
