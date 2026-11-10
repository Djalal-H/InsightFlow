"""Agent node: one ReAct step. The model reads the observations and chooses the next action.

The node only decides; `act` runs the action. It never loops by itself: the graph returns here
after each action, and the step budget, the repeated-call check, and the fallbacks below make
sure that every run either schedules one new action or `answer`.
"""

import json
from typing import Any

from insightflow.agents.document_search import SEARCH_SPEC
from insightflow.agents.graph import AgentNode
from insightflow.agents.models import ANSWER_ACTION, SEARCH_TOOL, AgentAction, Observation
from insightflow.agents.observations import render_observations, render_tool_catalog
from insightflow.agents.state import AgentState, attempt_observations, render_history
from insightflow.agents.structured import complete_json
from insightflow.providers.llm import ChatProvider
from insightflow.tools.models import ToolSpec
from insightflow.tools.registry import ToolSet

AGENT_SYSTEM_PROMPT = (
    "You are the research agent of InsightFlow, an assistant that answers questions from "
    "indexed documents and can use tools. Work in steps. In each step, return only a JSON "
    'object: {"thought": "<one short sentence>", "action": "<tool name or answer>", '
    '"arguments": {...}}. '
    "Use search_documents for facts that the indexed documents can hold, and search again with "
    "other words when the passages are not enough. Use the other tools as their descriptions "
    "say, for example to calculate with numbers from the passages. The arguments must match "
    "the tool's input schema. Do not repeat a call that already has a result. "
    'Use the action "answer" with no arguments when the observations are enough, when the '
    "message is a greeting or a question about the conversation, or when more steps will not "
    "help. Do not write the answer itself."
)

ANSWER = AgentAction(action=ANSWER_ACTION)


def make_agent_node(
    chat_provider: ChatProvider,
    tools: ToolSet,
    *,
    max_steps: int,
    history_turns: int,
) -> AgentNode:
    """Bind the agent model and the tool set; `max_steps` limits actions per attempt."""
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")

    async def agent_node(state: AgentState) -> AgentState:
        observations = state.get("observations", [])
        used = len(attempt_observations(state))
        if used >= max_steps:
            return {"pending_action": ANSWER}

        catalog = await catalog_specs(tools)
        prompt = agent_prompt(state, catalog, max_steps - used, history_turns)
        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        action = await complete_json(chat_provider, messages, AgentAction)
        if action is None:
            # Unreadable output: search the question as written once, then answer.
            if observations:
                return {"pending_action": ANSWER}
            action = AgentAction(
                thought="Search the documents for the question.",
                action=SEARCH_TOOL,
                arguments={"query": state["query"]},
            )
        if not action.is_answer and any(_same_call(action, item) for item in observations):
            return {"pending_action": ANSWER}
        return {"pending_action": action}

    return agent_node


async def catalog_specs(tools: ToolSet) -> list[ToolSpec]:
    """The local search tool first, then the MCP tools; the reserved name is not overridden."""
    return [SEARCH_SPEC, *(spec for spec in await tools.list_tools() if spec.name != SEARCH_TOOL)]


def agent_prompt(
    state: AgentState, catalog: list[ToolSpec], steps_left: int, history_turns: int
) -> str:
    parts: list[str] = []
    history = render_history(state, history_turns)
    if history:
        parts.append(f"Conversation so far:\n{history}")
    parts.append(f"Tools:\n{render_tool_catalog(catalog)}")
    parts.append(f"Question: {state['query']}")
    feedback = state.get("feedback", [])
    if feedback:
        issues = "\n".join(f"- {issue}" for issue in feedback)
        parts.append(
            "Your previous answer failed verification:\n"
            f"{issues}\nFind better evidence before you answer again."
        )
    parts.append(f"Observations so far:\n{render_observations(state.get('observations', []))}")
    parts.append(f"Steps left: {steps_left}")
    return "\n\n".join(parts)


def _same_call(action: AgentAction, observation: Observation) -> bool:
    """Compare with an earlier call, ignoring arguments that the tool set fixed."""
    result = observation.result
    if action.action != result.tool:
        return False
    return all(
        _canonical(result.arguments.get(key)) == _canonical(value)
        for key, value in action.arguments.items()
    )


def _canonical(value: Any) -> str:
    if isinstance(value, str):
        value = value.strip().lower()
    return json.dumps(value, sort_keys=True)
