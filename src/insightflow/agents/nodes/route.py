"""Route node: decides whether a question needs retrieval and makes it self-contained."""

from insightflow.agents.graph import AgentNode
from insightflow.agents.models import RouteDecision
from insightflow.agents.state import AgentState, render_history
from insightflow.agents.structured import complete_json
from insightflow.providers.llm import ChatProvider

ROUTER_SYSTEM_PROMPT = (
    "You route questions for an assistant that answers from indexed documents. "
    'Return only a JSON object: {"route": "retrieve" or "direct", "standalone_query": "..."}. '
    'Use "direct" only for greetings, thanks, questions about the assistant itself, or '
    "questions that the conversation so far answers completely. "
    'Use "retrieve" for all other questions. '
    "Write standalone_query as a self-contained version of the latest question: replace "
    "references to the conversation with the words they refer to, and keep the language of "
    "the question."
)


def make_route_node(chat_provider: ChatProvider | None, *, history_turns: int) -> AgentNode:
    """Bind the router model. With no provider, every question is retrieved as written."""

    async def route_node(state: AgentState) -> AgentState:
        query = state["query"]
        fallback: AgentState = {"route": "retrieve", "retrieval_query": query}
        if chat_provider is None:
            return fallback

        history = render_history(state, history_turns)
        conversation = f"Conversation so far:\n{history}\n\n" if history else ""
        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": f"{conversation}Latest question: {query}"},
        ]
        decision = await complete_json(chat_provider, messages, RouteDecision)
        if decision is None:
            return fallback
        return {"route": decision.route, "retrieval_query": decision.standalone_query or query}

    return route_node
