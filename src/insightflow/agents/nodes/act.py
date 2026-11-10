"""Act node: runs the agent's pending action and records the observation."""

from insightflow.agents.document_search import DocumentSearch
from insightflow.agents.graph import AgentNode
from insightflow.agents.models import SEARCH_TOOL, Observation
from insightflow.agents.state import AgentState
from insightflow.tools.registry import ToolSet


def make_act_node(search: DocumentSearch, tools: ToolSet) -> AgentNode:
    """Send `search_documents` to the local search and every other action to the tool set."""

    async def act_node(state: AgentState) -> AgentState:
        action = state.get("pending_action")
        if action is None or action.is_answer:
            raise ValueError("act needs a pending tool action")
        if action.action == SEARCH_TOOL:
            result, passages = await search.search(action.arguments)
        else:
            result, passages = await tools.call_tool(action.action, action.arguments), []
        observation = Observation(
            attempt=state.get("attempt", 1),
            thought=action.thought,
            result=result,
            passages=passages,
        )
        return {
            "observations": [*state.get("observations", []), observation],
            "pending_action": None,
        }

    return act_node
