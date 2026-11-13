"""LangGraph construction helpers.

The workflow is a ReAct loop with a verified answer::

    START → agent ⇄ act               (search_documents, calculate, tavily_search, ...)
              │
              └→ answer ─┬─ verify ─┬─ finalize → END
                         │          └─ retry → agent   (failed, attempts left)
                         └─ finalize → END             (no observations: direct reply)

With conversation memory, a `recall` node runs first (START → recall → agent).

Both loops are bounded. `agent` ⇄ `act` stops after `max_steps` actions per attempt (the agent
node enforces it, see `agents/nodes/agent.py`), and `retry` runs at most `max_attempts - 1` times.
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

NODE_NAMES = ("agent", "act", "answer", "verify", "retry", "finalize")


@dataclass(frozen=True)
class WorkflowNodes:
    """The node implementations wired into the workflow graph."""

    agent: AgentNode
    act: AgentNode
    answer: AgentNode
    verify: AgentNode
    retry: AgentNode
    finalize: AgentNode
    recall: AgentNode | None = None


def after_agent(state: AgentState) -> Literal["act", "answer"]:
    """Run the chosen tool, or answer when the agent chose `answer`."""
    action = state.get("pending_action")
    return "answer" if action is None or action.is_answer else "act"


def after_answer(state: AgentState) -> Literal["verify", "finalize"]:
    """Direct replies (no observations) are not verified."""
    return "verify" if state.get("observations") else "finalize"


def make_after_verify(max_attempts: int) -> Callable[[AgentState], Literal["retry", "finalize"]]:
    """Retry a failed verification until `max_attempts` answers were produced."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    def after_verify(state: AgentState) -> Literal["retry", "finalize"]:
        verification = state.get("verification")
        failed = verification is not None and verification.status == "failed"
        if failed and state.get("attempt", 1) < max_attempts:
            return "retry"
        return "finalize"

    return after_verify


def recursion_limit(max_steps: int, max_attempts: int) -> int:
    """The LangGraph step limit that the bounded loops can never reach, plus a margin."""
    per_attempt = 2 * max_steps + 3  # agent/act pairs, the final agent, answer, and verify
    # One retry per attempt, then recall, finalize, and a margin.
    return max_attempts * (per_attempt + 1) + 3


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
    """Build the ReAct workflow with verification and bounded retries."""
    graph = StateGraph(AgentState)
    for name in NODE_NAMES:
        graph.add_node(name, traced(name, getattr(nodes, name)))  # type: ignore[call-overload]
    if nodes.recall is not None:
        graph.add_node("recall", traced("recall", nodes.recall))  # type: ignore[call-overload]
        graph.add_edge(START, "recall")
        graph.add_edge("recall", "agent")
    else:
        graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", after_agent)
    graph.add_edge("act", "agent")
    graph.add_conditional_edges("answer", after_answer)
    graph.add_conditional_edges("verify", make_after_verify(max_attempts))
    graph.add_edge("retry", "agent")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)
