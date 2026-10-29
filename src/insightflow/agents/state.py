"""Shared LangGraph workflow state."""

from typing import TypedDict

from insightflow.rag.models import AssembledContext


class AgentState(TypedDict, total=False):
    """State passed between the retrieval and answer nodes."""

    query: str
    context: AssembledContext
    answer: str
