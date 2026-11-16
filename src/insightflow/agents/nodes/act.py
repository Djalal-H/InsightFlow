"""Act node: runs the agent's pending action and records the observation."""

from insightflow.agents.document_search import DocumentSearch
from insightflow.agents.graph import AgentNode
from insightflow.agents.models import SEARCH_TOOL, Observation
from insightflow.agents.nodes.approve import NOT_AUTHORIZED_OUTPUT, approval_id_for
from insightflow.agents.state import AgentState
from insightflow.core.observability import record_tool_call
from insightflow.guardrails.untrusted import WebAction, screen_observation
from insightflow.security.audit import audit
from insightflow.tools.registry import ToolSet, refused_result


def make_act_node(
    search: DocumentSearch, tools: ToolSet, *, web_action: WebAction = "drop"
) -> AgentNode:
    """Send `search_documents` to the local search and every other action to the tool set.

    The result is screened before it is recorded (`guardrails/untrusted.py`); signals go to
    `guard_signals`.
    """

    async def act_node(state: AgentState) -> AgentState:
        action = state.get("pending_action")
        if action is None or action.is_answer:
            raise ValueError("act needs a pending tool action")
        if action.action == SEARCH_TOOL:
            result, passages = await search.search(
                action.arguments, scope=state.get("search_scope")
            )
        elif needs_approval(tools, action.action) and state.get(
            "approved_call"
        ) != approval_id_for(state, action):
            # Defense in depth: only the approve node can let a consequential call through.
            audit("tool_call_refused", tool=action.action, reason="not_approved")
            result, passages = (
                refused_result(
                    action.action,
                    action.arguments,
                    "not_authorized",
                    NOT_AUTHORIZED_OUTPUT,
                    tools.server_of(action.action),
                ),
                [],
            )
        else:
            result, passages = await tools.call_tool(action.action, action.arguments), []
        record_tool_call(
            tool=result.tool,
            server=result.server,
            started_at=result.started_at,
            duration_ms=result.duration_ms,
            status=result.status,
            error=result.error,
        )
        result, signals = screen_observation(result, passages, web_action=web_action)
        observation = Observation(
            attempt=state.get("attempt", 1),
            thought=action.thought,
            result=result,
            passages=passages,
        )
        update: AgentState = {
            "observations": [*state.get("observations", []), observation],
            "pending_action": None,
            "pending_consequential": False,
            "approved_call": None,
        }
        if signals:
            update["guard_signals"] = [*state.get("guard_signals", []), *signals]
        return update

    return act_node


def needs_approval(tools: ToolSet, name: str) -> bool:
    """A listed tool that is not read-only; unknown tools fail in the tool set instead."""
    return bool(tools.server_of(name)) and tools.is_consequential(name)
