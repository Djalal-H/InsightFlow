"""Recall node: finds earlier turns of the thread that are relevant to the new question."""

import logging

from insightflow.agents.graph import AgentNode
from insightflow.agents.memory import ConversationMemory
from insightflow.agents.state import AgentState
from insightflow.core.exceptions import InsightFlowError

logger = logging.getLogger(__name__)


def make_recall_node(memory: ConversationMemory, *, history_turns: int) -> AgentNode:
    """Recall only turns that are older than the `history_turns` window shown in prompts.

    Memory is an aid, not a requirement: a storage or provider failure is logged and the turn
    continues without recalled turns.
    """

    async def recall_node(state: AgentState) -> AgentState:
        thread_id = state.get("thread_id")
        before_turn = len(state.get("history", [])) - max(history_turns, 0)
        if not thread_id or before_turn <= 0:
            return {"recalled": []}
        try:
            recalled = await memory.recall(thread_id, state["query"], before_turn=before_turn)
        except InsightFlowError as exc:
            logger.warning("Conversation memory recall failed: %s", type(exc).__name__)
            recalled = []
        return {"recalled": recalled}

    return recall_node
