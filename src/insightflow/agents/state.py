"""Shared LangGraph workflow state."""

import operator
from typing import Annotated, TypedDict

from insightflow.agents.models import ConversationTurn, Route, VerificationResult
from insightflow.providers.llm import ChatMessage
from insightflow.rag.models import AssembledContext


class AgentState(TypedDict, total=False):
    """State passed between workflow nodes.

    `history` persists across the turns of a checkpointed thread; nodes append to it. Every other
    field belongs to one turn and is reset by `new_turn_input`, because a checkpointer keeps the
    values of the previous turn.
    """

    history: Annotated[list[ConversationTurn], operator.add]
    query: str
    route: Route | None
    retrieval_query: str | None
    retrieval_queries: list[str]
    context: AssembledContext | None
    answer: str
    verification: VerificationResult | None
    attempt: int
    node_trace: list[str]


def new_turn_input(query: str) -> AgentState:
    """Return the graph input for a new turn, with every per-turn field reset."""
    return {
        "query": query,
        "route": None,
        "retrieval_query": None,
        "retrieval_queries": [],
        "context": None,
        "answer": "",
        "verification": None,
        "attempt": 1,
        "node_trace": [],
    }


def history_messages(state: AgentState, max_turns: int) -> list[ChatMessage]:
    """Render the last `max_turns` turns of the conversation as chat messages."""
    if max_turns <= 0:
        return []
    messages: list[ChatMessage] = []
    for turn in state.get("history", [])[-max_turns:]:
        messages.append({"role": "user", "content": turn.query})
        messages.append({"role": "assistant", "content": turn.answer})
    return messages


def render_history(state: AgentState, max_turns: int) -> str:
    """Render recent turns as plain text for prompts that judge or rewrite a question."""
    if max_turns <= 0:
        return ""
    lines = [
        f"User: {turn.query}\nAssistant: {turn.answer}"
        for turn in state.get("history", [])[-max_turns:]
    ]
    return "\n\n".join(lines)
