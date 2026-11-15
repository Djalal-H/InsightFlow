"""Approve node: a consequential tool call waits for the user's decision.

The agent node marks an action as consequential when the tool set does not list the tool as
read-only. The graph then routes it here instead of to `act`:

- Without the `tools:consequential` scope, or without a checkpointer (a paused run must be
  stored to resume), the call is refused with a `not_authorized` observation.
- Otherwise the run pauses with LangGraph's `interrupt`. The API returns the exact tool,
  server, and arguments with an approval ID, and the client resumes the run with a decision.
  An approval counts only for that ID, so it cannot approve a different call.

LangGraph runs the node again from its start on resume, so everything before `interrupt` is
deterministic: the approval ID is derived from the turn's state and the call.
"""

import hashlib
import json
from typing import Any

from langgraph.errors import GraphInterrupt
from langgraph.types import interrupt

from insightflow.agents.graph import AgentNode
from insightflow.agents.models import AgentAction, Observation
from insightflow.agents.state import AgentState
from insightflow.core.observability import register_error_code
from insightflow.security.audit import audit
from insightflow.security.auth import ADMIN, TOOLS_CONSEQUENTIAL
from insightflow.tools.registry import ToolSet, refused_result

register_error_code(GraphInterrupt, "paused")

NOT_AUTHORIZED_OUTPUT = (
    "This tool changes data or sends something, and the user may not approve such calls here."
)
NOT_APPROVED_OUTPUT = "The user rejected this call. Do not call it again in this turn."


def approval_id_for(state: AgentState, action: AgentAction) -> str:
    """A stable identifier of this call in this turn: same state and call, same ID."""
    material = json.dumps(
        {
            "thread": state.get("thread_id"),
            "query": state.get("query", ""),
            "attempt": state.get("attempt", 1),
            "step": len(state.get("observations", [])),
            "tool": action.action,
            "arguments": action.arguments,
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


def may_use_consequential_tools(state: AgentState) -> bool:
    scopes = state.get("granted_scopes", [])
    return TOOLS_CONSEQUENTIAL in scopes or ADMIN in scopes


def approval_request(state: AgentState, action: AgentAction, server: str) -> dict[str, Any]:
    """The interrupt value: what the user approves, exactly."""
    return {
        "approval_id": approval_id_for(state, action),
        "tool": action.action,
        "server": server,
        "arguments": dict(action.arguments),
        "thought": action.thought,
    }


def make_approve_node(tools: ToolSet, *, can_pause: bool) -> AgentNode:
    """Bind the tool set; `can_pause` is true only with a checkpointer."""

    def refuse(state: AgentState, action: AgentAction, code: str, output: str) -> AgentState:
        result = refused_result(
            action.action,
            action.arguments,
            "not_authorized" if code == "not_authorized" else "not_approved",
            output,
            tools.server_of(action.action),
        )
        observation = Observation(
            attempt=state.get("attempt", 1), thought=action.thought, result=result
        )
        return {
            "observations": [*state.get("observations", []), observation],
            "pending_action": None,
            "pending_consequential": False,
            "approved_call": None,
        }

    async def approve_node(state: AgentState) -> AgentState:
        action = state.get("pending_action")
        if action is None or action.is_answer:
            raise ValueError("approve needs a pending tool action")
        if not may_use_consequential_tools(state) or not can_pause:
            reason = "no_checkpointer" if may_use_consequential_tools(state) else "missing_scope"
            audit("tool_call_refused", tool=action.action, reason=reason)
            return refuse(state, action, "not_authorized", NOT_AUTHORIZED_OUTPUT)

        request = approval_request(state, action, tools.server_of(action.action))
        audit("approval_requested", tool=action.action, approval_id=request["approval_id"])
        decision = interrupt(request)
        approved = (
            isinstance(decision, dict)
            and decision.get("approval_id") == request["approval_id"]
            and decision.get("approved") is True
        )
        audit(
            "approval_decided",
            tool=action.action,
            approval_id=request["approval_id"],
            approved=approved,
        )
        if not approved:
            return refuse(state, action, "not_approved", NOT_APPROVED_OUTPUT)
        return {"approved_call": request["approval_id"]}

    return approve_node
