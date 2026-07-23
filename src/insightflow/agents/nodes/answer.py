"""Answer node: produces a response using an injected chat provider."""

from insightflow.agents.graph import AgentNode
from insightflow.agents.state import AgentState
from insightflow.providers.llm import ChatProvider


def make_answer_node(chat_provider: ChatProvider) -> AgentNode:
    """Bind a chat provider and return a LangGraph-compatible answer node."""

    async def answer_node(state: AgentState) -> AgentState:
        query = state.get("query", "")
        messages = [{"role": "user", "content": query}]
        answer = await chat_provider.complete(messages)
        return {
            **state,
            "answer": answer,
            "sources": state.get("sources", []),
        }

    return answer_node
