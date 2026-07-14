"""Shared LangGraph workflow state."""

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """State passed between initial workflow nodes."""

    query: str
    messages: list[dict[str, Any]]
    retrieved_context: list[dict[str, Any]]
    answer: str
    sources: list[str]
    error: str

