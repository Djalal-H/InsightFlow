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

    `search_scope` holds the request's document filters (`file_types`, `sources`); `recalled`
    holds older turns that conversation memory found for this question.
    """

    history: Annotated[list[ConversationTurn], operator.add]
    query: str
    thread_id: str | None
    search_scope: dict[str, list[str]]
    recalled: list[ConversationTurn]
    observations: list[Observation]
    pending_action: AgentAction | None
    answer: str
    verification: VerificationResult | None
    feedback: list[str]
    attempt: int
    node_trace: list[str]


def new_turn_input(
    query: str,
    *,
    thread_id: str | None = None,
    search_scope: dict[str, list[str]] | None = None,
) -> AgentState:
    """Return the graph input for a new turn, with every per-turn field reset."""
    return {
        "query": query,
        "thread_id": thread_id,
        "search_scope": dict(search_scope or {}),
        "recalled": [],
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


def prompt_turns(state: AgentState, max_turns: int) -> list[ConversationTurn]:
    """The turns shown in prompts: recalled older turns, then the last `max_turns` turns."""
    recent = state.get("history", [])[-max_turns:] if max_turns > 0 else []
    return [*state.get("recalled", []), *recent]


def history_messages(state: AgentState, max_turns: int) -> list[ChatMessage]:
    """Render the recalled and the last `max_turns` turns as chat messages."""
    messages: list[ChatMessage] = []
    for turn in prompt_turns(state, max_turns):
        messages.append({"role": "user", "content": turn.query})
        messages.append({"role": "assistant", "content": turn.answer})
    return messages


def render_history(state: AgentState, max_turns: int) -> str:
    """Render the same turns as plain text for prompts that plan or judge."""
    return "\n\n".join(
        f"User: {turn.query}\nAssistant: {turn.answer}"
        for turn in prompt_turns(state, max_turns)
    )
