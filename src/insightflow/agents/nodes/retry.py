"""Retry node: starts the next attempt after a failed verification."""

from insightflow.agents.state import AgentState


async def retry_node(state: AgentState) -> AgentState:
    """Give the verification issues to the agent and reset the answer for the next attempt."""
    verification = state.get("verification")
    issues = list(verification.issues) if verification is not None else []
    return {
        "attempt": state.get("attempt", 1) + 1,
        "feedback": issues or ["The previous answer was not supported by the observations."],
        "answer": "",
        "verification": None,
    }
