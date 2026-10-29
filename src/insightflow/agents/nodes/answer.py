"""Answer node: produces a grounded response using an injected chat provider."""

from insightflow.agents.graph import AgentNode
from insightflow.agents.state import AgentState
from insightflow.providers.llm import ChatProvider

NO_CONTEXT_ANSWER = "I could not find relevant information in the indexed documents."
GROUNDED_SYSTEM_PROMPT = (
    "Answer the question using only the numbered context passages. "
    "Cite every passage you use by its number in square brackets, for example [1]. "
    "If the passages do not contain the answer, say that you do not know."
)


def make_answer_node(chat_provider: ChatProvider) -> AgentNode:
    """Bind a chat provider and return a LangGraph-compatible answer node."""

    async def answer_node(state: AgentState) -> AgentState:
        context = state.get("context")
        if context is None or not context.chunks:
            return {"answer": NO_CONTEXT_ANSWER}
        messages = [
            {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context.rendered}\n\nQuestion: {state['query']}",
            },
        ]
        return {"answer": await chat_provider.complete(messages)}

    return answer_node
