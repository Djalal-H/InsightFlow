"""Answer node: produces a grounded or direct response using an injected chat provider."""

from collections.abc import Sequence

from langgraph.config import get_stream_writer

from insightflow.agents.graph import AgentNode
from insightflow.agents.state import AgentState, history_messages
from insightflow.providers.llm import ChatMessage, ChatProvider

NO_CONTEXT_ANSWER = "I could not find relevant information in the indexed documents."
GROUNDED_SYSTEM_PROMPT = (
    "Answer the question using only the numbered context passages. "
    "Cite every passage you use by its number in square brackets, for example [1]. "
    "If the passages do not contain the answer, say that you do not know."
)
DIRECT_SYSTEM_PROMPT = (
    "You are InsightFlow, an assistant that answers questions from indexed documents. "
    "Reply briefly to the latest message. Use only the conversation so far. Do not state facts "
    "that are not in the conversation; offer to search the documents instead."
)


def make_answer_node(
    chat_provider: ChatProvider,
    *,
    history_turns: int = 0,
    stream_tokens: bool = False,
) -> AgentNode:
    """Bind a chat provider and return a LangGraph-compatible answer node.

    With `stream_tokens`, text is generated with `ChatProvider.stream` and every piece is sent to
    LangGraph's custom stream as `{"type": "token", "attempt": n, "text": ...}`.
    """

    async def generate(messages: Sequence[ChatMessage], attempt: int) -> str:
        if not stream_tokens:
            return await chat_provider.complete(messages)
        write = get_stream_writer()
        pieces: list[str] = []
        async for piece in chat_provider.stream(messages):
            pieces.append(piece)
            write({"type": "token", "attempt": attempt, "text": piece})
        return "".join(pieces)

    async def answer_node(state: AgentState) -> AgentState:
        query = state["query"]
        attempt = state.get("attempt", 1)
        history = history_messages(state, history_turns)

        if state.get("route") == "direct":
            messages: list[ChatMessage] = [
                {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
                *history,
                {"role": "user", "content": query},
            ]
            return {"answer": await generate(messages, attempt)}

        context = state.get("context")
        if context is None or not context.chunks:
            if stream_tokens:
                get_stream_writer()(
                    {"type": "token", "attempt": attempt, "text": NO_CONTEXT_ANSWER}
                )
            return {"answer": NO_CONTEXT_ANSWER}

        messages = [
            {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
            *history,
            {
                "role": "user",
                "content": f"Context:\n{context.rendered}\n\nQuestion: {query}",
            },
        ]
        return {"answer": await generate(messages, attempt)}

    return answer_node
