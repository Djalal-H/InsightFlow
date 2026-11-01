"""Rewrite node: produces a new retrieval query after a failed verification."""

from insightflow.agents.graph import AgentNode
from insightflow.agents.models import RewrittenQuery
from insightflow.agents.state import AgentState
from insightflow.agents.structured import complete_json
from insightflow.providers.llm import ChatProvider

REWRITE_SYSTEM_PROMPT = (
    "You improve search queries for a document retrieval system. The previous searches did not "
    "find passages that support a good answer. "
    'Return only a JSON object: {"query": "..."}. '
    "Use different keywords, synonyms, or a more specific phrase, and keep the meaning and the "
    "language of the question."
)


def make_rewrite_node(chat_provider: ChatProvider) -> AgentNode:
    """Bind the rewrite model and return a node that starts the next attempt."""

    async def rewrite_node(state: AgentState) -> AgentState:
        question = state["query"]
        previous = state.get("retrieval_queries", [])
        verification = state.get("verification")
        issues = verification.issues if verification is not None else []
        prompt = "\n".join(
            [
                f"Question: {question}",
                "Previous search queries:",
                *(f"- {query}" for query in previous),
                "Problems with the previous answer:",
                *(f"- {issue}" for issue in issues),
            ]
        )
        messages = [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        rewritten = await complete_json(chat_provider, messages, RewrittenQuery)
        query = rewritten.query if rewritten is not None and rewritten.query else question
        return {"retrieval_query": query, "attempt": state.get("attempt", 1) + 1}

    return rewrite_node
