"""Shared LangGraph workflow state."""

import operator
from typing import Annotated, TypedDict

from insightflow.agents.models import (
    AgentAction,
    ConversationTurn,
    Observation,
    VerificationResult,
)
from insightflow.providers.llm import ChatMessage


class AgentState(TypedDict, total=False):
    """State passed between workflow nodes.

    `history` persists across the turns of a checkpointed thread; nodes append to it. Every other
    field belongs to one turn and is reset by `new_turn_input`, because a checkpointer keeps the
    values of the previous turn.
    """

    history: Annotated[list[ConversationTurn], operator.add]
    query: str
    observations: list[Observation]
    pending_action: AgentAction | None
    answer: str
    verification: VerificationResult | None
    feedback: list[str]
    attempt: int
    node_trace: list[str]


def new_turn_input(query: str) -> AgentState:
    """Return the graph input for a new turn, with every per-turn field reset."""
    return {
        "query": query,
        "observations": [],
        "pending_action": None,
        "answer": "",
        "verification": None,
        "feedback": [],
        "attempt": 1,
        "node_trace": [],
    }


def attempt_observations(state: AgentState) -> list[Observation]:
    """Return the observations made in the current attempt."""
    attempt = state.get("attempt", 1)
    return [item for item in state.get("observations", []) if item.attempt == attempt]


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
    """Render recent turns as plain text for prompts that plan or judge."""
    if max_turns <= 0:
        return ""
    lines = [
        f"User: {turn.query}\nAssistant: {turn.answer}"
        for turn in state.get("history", [])[-max_turns:]
    ]
    return "\n\n".join(lines)
