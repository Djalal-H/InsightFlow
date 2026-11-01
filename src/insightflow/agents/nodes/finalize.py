"""Finalize node: records the turn in the conversation history."""

from insightflow.agents.models import ConversationTurn, VerificationResult
from insightflow.agents.state import AgentState


async def finalize_node(state: AgentState) -> AgentState:
    """Append the completed turn and mark unverified (direct) answers as skipped."""
    update: AgentState = {
        "history": [ConversationTurn(query=state["query"], answer=state.get("answer", ""))],
    }
    if state.get("verification") is None:
        update["verification"] = VerificationResult(status="skipped", method="none")
    return update
