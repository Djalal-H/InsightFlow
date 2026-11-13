"""Finalize node: records the turn in the conversation history and, optionally, in memory."""

import logging

from insightflow.agents.graph import AgentNode
from insightflow.agents.memory import ConversationMemory
from insightflow.agents.models import ConversationTurn, VerificationResult
from insightflow.agents.state import AgentState
from insightflow.core.exceptions import InsightFlowError

logger = logging.getLogger(__name__)


async def finalize_node(state: AgentState) -> AgentState:
    """Append the completed turn and mark unverified (direct) answers as skipped."""
    update: AgentState = {
        "history": [ConversationTurn(query=state["query"], answer=state.get("answer", ""))],
    }
    if state.get("verification") is None:
        update["verification"] = VerificationResult(status="skipped", method="none")
    return update


def make_finalize_node(memory: ConversationMemory | None = None) -> AgentNode:
    """Return the finalize node; with `memory`, the turn of a thread is also remembered.

    A memory failure is logged and does not fail the turn: the answer is already written.
    """
    if memory is None:
        return finalize_node

    async def remembering_finalize_node(state: AgentState) -> AgentState:
        update = await finalize_node(state)
        thread_id = state.get("thread_id")
        if thread_id:
            try:
                await memory.remember(
                    thread_id, len(state.get("history", [])), update["history"][0]
                )
            except InsightFlowError as exc:
                logger.warning("Conversation memory write failed: %s", type(exc).__name__)
        return update

    return remembering_finalize_node
