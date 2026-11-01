"""LangGraph construction helpers.

The workflow is::

    START → route ─┬─ direct ───────────────────────────→ answer → finalize → END
                   └─ retrieve → retrieve → answer → verify ─┬─ finalize → END
                                    ▲                        │
                                    └────── rewrite ◄────────┘ (failed, attempts left)
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from insightflow.agents.state import AgentState

AgentNode = Callable[[AgentState], Awaitable[AgentState]]
AgentGraph = CompiledStateGraph[AgentState, None, AgentState, AgentState]


@dataclass(frozen=True)
class WorkflowNodes:
    """The node implementations wired into the workflow graph."""

    route: AgentNode
    retrieve: AgentNode
    answer: AgentNode
    verify: AgentNode
    rewrite: AgentNode
    finalize: AgentNode


def after_route(state: AgentState) -> Literal["retrieve", "answer"]:
    """Skip retrieval for questions the router marked as direct."""
    return "answer" if state.get("route") == "direct" else "retrieve"


def after_answer(state: AgentState) -> Literal["verify", "finalize"]:
    """Only grounded answers are verified."""
    return "finalize" if state.get("route") == "direct" else "verify"


def make_after_verify(max_attempts: int) -> Callable[[AgentState], Literal["rewrite", "finalize"]]:
    """Retry a failed verification until `max_attempts` answers were produced."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    def after_verify(state: AgentState) -> Literal["rewrite", "finalize"]:
        verification = state.get("verification")
        failed = verification is not None and verification.status == "failed"
        if failed and state.get("attempt", 1) < max_attempts:
            return "rewrite"
        return "finalize"

    return after_verify


def traced(name: str, node: AgentNode) -> AgentNode:
    """Wrap a node so that its name is appended to the turn's `node_trace`."""

    async def traced_node(state: AgentState) -> AgentState:
        update = await node(state)
        update["node_trace"] = [*state.get("node_trace", []), name]
        return update

    return traced_node


def build_graph(
    nodes: WorkflowNodes,
    *,
    max_attempts: int = 1,
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> AgentGraph:
    """Build the routed, verified answer workflow."""
    graph = StateGraph(AgentState)
    for name in ("route", "retrieve", "answer", "verify", "rewrite", "finalize"):
        graph.add_node(name, traced(name, getattr(nodes, name)))  # type: ignore[call-overload]
    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", after_route)
    graph.add_edge("retrieve", "answer")
    graph.add_conditional_edges("answer", after_answer)
    graph.add_conditional_edges("verify", make_after_verify(max_attempts))
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)
