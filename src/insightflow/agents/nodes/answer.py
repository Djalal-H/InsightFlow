"""Answer node: writes the final answer from the agent's observations, or a direct reply."""

from collections.abc import Sequence

from langgraph.config import get_stream_writer

from insightflow.agents.graph import AgentNode
from insightflow.agents.observations import has_evidence, render_observations
from insightflow.agents.state import AgentState, history_messages
from insightflow.guardrails.classifier import UNSAFE_ANSWER, UNSAFE_SIGNAL, SafetyClassifier
from insightflow.guardrails.output import AnswerGuard, canary_instruction, new_canary
from insightflow.guardrails.spotlight import UNTRUSTED_RULE
from insightflow.providers.llm import ChatMessage, ChatProvider
from insightflow.security.audit import audit

NO_CONTEXT_ANSWER = "I could not find relevant information in the indexed documents."
GROUNDED_SYSTEM_PROMPT = (
    "Answer the question using only the observations: numbered sources and tool results. "
    "Cite every source you use by its number in square brackets, for example [1]. "
    "Give calculated values exactly as the tool returned them. "
    "If the observations do not contain the answer, say that you do not know. "
    + UNTRUSTED_RULE
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
    guard: AnswerGuard | None = None,
    safety: SafetyClassifier | None = None,
    datamark_web: bool = False,
) -> AgentNode:
    """Bind a chat provider and return a LangGraph-compatible answer node.

    With `stream_tokens`, text is generated with `ChatProvider.stream` and every piece is sent to
    LangGraph's custom stream as `{"type": "token", "attempt": n, "text": ...}`.

    Each generated answer goes through the output checks: a per-answer canary is added to the
    system prompt, and `guard` (default: no secrets, default size) removes exfiltration links,
    redacts secrets, and replaces a leaked prompt. `safety`, when given, classifies the answer.
    When a check changes a streamed answer, `{"type": "replace", "attempt": n}` tells the
    client that the tokens of this attempt are void and the final answer replaces them.
    """
    answer_guard = guard or AnswerGuard()

    async def generate(messages: Sequence[ChatMessage], attempt: int) -> str:
        if not stream_tokens:
            return await chat_provider.complete(messages)
        write = get_stream_writer()
        pieces: list[str] = []
        async for piece in chat_provider.stream(messages):
            pieces.append(piece)
            write({"type": "token", "attempt": attempt, "text": piece})
        return "".join(pieces)

    async def checked(
        state: AgentState, system_prompt: str, messages: list[ChatMessage]
    ) -> AgentState:
        attempt = state.get("attempt", 1)
        canary = new_canary()
        prompt = {"role": "system", "content": system_prompt + canary_instruction(canary)}
        raw = await generate([prompt, *messages], attempt)
        allowed = [
            page.url
            for observation in state.get("observations", [])
            for page in observation.result.web_sources
        ]
        result = answer_guard.check(raw, allowed_urls=allowed, canary=canary)
        text, signals = result.text, list(result.signals)
        if safety is not None and text == raw:
            verdict = await safety.check_output(state["query"], text)
            if not verdict.safe:
                text, signals = UNSAFE_ANSWER, [*signals, UNSAFE_SIGNAL]
        if signals:
            audit("answer_guarded", signals=signals)
            if stream_tokens and text != raw:
                get_stream_writer()({"type": "replace", "attempt": attempt})
        return {
            "answer": text,
            "guard_signals": [*state.get("guard_signals", []), *signals],
        }

    async def answer_node(state: AgentState) -> AgentState:
        query = state["query"]
        attempt = state.get("attempt", 1)
        history = history_messages(state, history_turns)
        observations = state.get("observations", [])

        if not observations:
            return await checked(
                state, DIRECT_SYSTEM_PROMPT, [*history, {"role": "user", "content": query}]
            )

        if not has_evidence(observations):
            if stream_tokens:
                get_stream_writer()(
                    {"type": "token", "attempt": attempt, "text": NO_CONTEXT_ANSWER}
                )
            return {"answer": NO_CONTEXT_ANSWER}

        rendered = render_observations(observations, datamark_web=datamark_web)
        return await checked(
            state,
            GROUNDED_SYSTEM_PROMPT,
            [
                *history,
                {"role": "user", "content": f"Observations:\n{rendered}\n\nQuestion: {query}"},
            ],
        )

    return answer_node
