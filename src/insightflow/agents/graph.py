"""LangGraph construction helpers."""

from collections.abc import Callable

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from insightflow.agents.state import AgentState

AgentNode = Callable[[AgentState], AgentState]


def build_graph(answer_node: AgentNode) -> CompiledStateGraph:
    """Build the smallest executable graph while later workflow nodes are developed."""
    graph = StateGraph(AgentState)
    graph.add_node("answer", answer_node)
    graph.add_edge(START, "answer")
    graph.add_edge("answer", END)
    return graph.compile()

