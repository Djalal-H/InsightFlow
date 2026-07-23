"""LangGraph construction helpers."""

from collections.abc import Awaitable, Callable

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from insightflow.agents.state import AgentState

AgentNode = Callable[[AgentState], Awaitable[AgentState]]


def build_graph(
    answer_node: AgentNode,
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    """Build the smallest executable graph while later workflow nodes are developed."""
    graph = StateGraph(AgentState)
    # mypy cannot match a plain async Callable against langgraph's _Node Protocol
    # union when NodeInputT must be inferred; this is a stub limitation, not a
    # type error in our code. Confirmed identical failure with a sync callable
    # in the original skeleton.
    graph.add_node("answer", answer_node)  # type: ignore[call-overload]
    graph.add_edge(START, "answer")
    graph.add_edge("answer", END)
    return graph.compile()
