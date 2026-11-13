"""Tests for retrieval-based conversation memory, its Qdrant store, and its graph nodes."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from qdrant_client import AsyncQdrantClient

from insightflow.agents.checkpoint import create_checkpoint_serializer, thread_config
from insightflow.agents.document_search import DocumentSearch
from insightflow.agents.graph import WorkflowNodes, build_graph, recursion_limit
from insightflow.agents.memory import ConversationMemory, memory_text
from insightflow.agents.models import ConversationTurn
from insightflow.agents.nodes.act import make_act_node
from insightflow.agents.nodes.agent import make_agent_node
from insightflow.agents.nodes.answer import make_answer_node
from insightflow.agents.nodes.finalize import make_finalize_node
from insightflow.agents.nodes.recall import make_recall_node
from insightflow.agents.nodes.retry import retry_node
from insightflow.agents.nodes.verify import make_verify_node
from insightflow.agents.state import AgentState, history_messages, new_turn_input, render_history
from insightflow.api.routes.query import node_event
from insightflow.core.config import Settings
from insightflow.core.exceptions import ProviderError, StorageError
from insightflow.rag.models import ContextBudget
from insightflow.rag.retrieval import BudgetedContextAssembler
from insightflow.storage.qdrant_memory import QdrantMemoryStore, memory_point_id
from insightflow.tools.registry import ToolSet
from tests.fakes import ANSWER, ScriptedChatProvider, StaticRetriever, word_count

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")

TOPICS = ("visa", "library", "tuition")


class TopicEmbeddingProvider:
    """Embeds text as topic-word counts; records what was embedded."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.texts: list[str] = []

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if self.error is not None:
            raise self.error
        self.texts.extend(texts)
        return [[float(text.lower().count(word)) + 0.01 for word in TOPICS] for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]

    def count_tokens(self, text: str) -> int:
        return max(1, len(text.split()))


@pytest.fixture
async def store() -> AsyncIterator[QdrantMemoryStore]:
    settings = Settings(_env_file=None, memory_collection="test_memory")
    memory_store = QdrantMemoryStore(settings, client=AsyncQdrantClient(location=":memory:"))
    yield memory_store
    await memory_store.close()


def make_memory(store: QdrantMemoryStore, **options: object) -> ConversationMemory:
    return ConversationMemory(TopicEmbeddingProvider(), store, **options)  # type: ignore[arg-type]


TURNS = [
    ConversationTurn(query="What visa do I need?", answer="A student visa [1]."),
    ConversationTurn(query="When is the library open?", answer="The library opens at 8 [1]."),
    ConversationTurn(query="How much is tuition?", answer="Tuition is 9000 [1]."),
]


async def remember_all(memory: ConversationMemory, thread_id: str = "t1") -> None:
    for index, turn in enumerate(TURNS):
        await memory.remember(thread_id, index, turn)


# --- memory and store --------------------------------------------------------------------


async def test_recall_returns_the_most_similar_older_turns_oldest_first(
    store: QdrantMemoryStore,
) -> None:
    memory = make_memory(store, top_k=2)
    await remember_all(memory)

    recalled = await memory.recall("t1", "tuition and visa costs", before_turn=3)
    only_first = await memory.recall("t1", "tuition and visa costs", before_turn=1)

    assert recalled == [TURNS[0], TURNS[2]]
    assert only_first == [TURNS[0]]


async def test_recall_is_separated_by_thread_and_respects_the_threshold(
    store: QdrantMemoryStore,
) -> None:
    await remember_all(make_memory(store))

    other_thread = await make_memory(store).recall("t2", "visa", before_turn=3)
    strict = await make_memory(store, score_threshold=0.9).recall("t1", "visa", before_turn=3)

    assert other_thread == []
    assert strict == [TURNS[0]]


async def test_recall_before_any_write_and_before_turn_zero_returns_nothing(
    store: QdrantMemoryStore,
) -> None:
    memory = make_memory(store)

    assert await memory.recall("t1", "visa", before_turn=5) == []
    await remember_all(memory)
    assert await memory.recall("t1", "visa", before_turn=0) == []


async def test_remembering_a_position_again_replaces_it(store: QdrantMemoryStore) -> None:
    memory = make_memory(store)
    await memory.remember("t1", 0, TURNS[0])
    await memory.remember("t1", 0, ConversationTurn(query="visa?", answer="A work visa."))
    await memory.remember("t1", 1, ConversationTurn(query="visa?", answer="  "))

    recalled = await memory.recall("t1", "visa", before_turn=9)

    assert [turn.answer for turn in recalled] == ["A work visa."]
    assert memory_point_id("t1", 0) == memory_point_id("t1", 0) != memory_point_id("t1", 1)


async def test_forget_deletes_only_the_thread(store: QdrantMemoryStore) -> None:
    memory = make_memory(store)
    await memory.forget("t1")  # no collection yet: nothing to do
    await remember_all(memory, "t1")
    await remember_all(memory, "t2")

    await memory.forget("t1")

    assert await memory.recall("t1", "visa", before_turn=3) == []
    assert await memory.recall("t2", "visa", before_turn=3) != []


async def test_store_failures_are_storage_errors() -> None:
    client = AsyncQdrantClient(location=":memory:")
    store = QdrantMemoryStore(Settings(_env_file=None), client=client)
    await client.close()

    with pytest.raises(StorageError, match="memory write"):
        await store.save_turn("t1", 0, [1.0], query="q", answer="a")
    with pytest.raises(StorageError, match="memory search"):
        await store.search_turns("t1", [1.0], limit=1, before_turn=1)
    with pytest.raises(StorageError, match="memory delete"):
        await store.delete_thread("t1")


def test_memory_text_is_clipped() -> None:
    assert memory_text("q", "a") == "User: q\nAssistant: a"
    assert len(memory_text("q" * 5000, "a")) == 4000
    with pytest.raises(ValueError, match="top_k"):
        ConversationMemory(TopicEmbeddingProvider(), None, top_k=0)  # type: ignore[arg-type]


# --- nodes -------------------------------------------------------------------------------


def state_with_history(query: str, *, thread_id: str | None = "t1") -> AgentState:
    state = new_turn_input(query, thread_id=thread_id)
    state["history"] = list(TURNS)
    return state


async def test_recall_node_finds_turns_outside_the_prompt_window(
    store: QdrantMemoryStore,
) -> None:
    memory = make_memory(store, top_k=1)
    await remember_all(memory)
    node = make_recall_node(memory, history_turns=1)

    update = await node(state_with_history("What about my visa?"))

    assert update == {"recalled": [TURNS[0]]}


async def test_recall_node_skips_stateless_requests_and_short_threads(
    store: QdrantMemoryStore,
) -> None:
    memory = make_memory(store)
    await remember_all(memory)

    stateless = await make_recall_node(memory, history_turns=1)(
        state_with_history("visa?", thread_id=None)
    )
    in_window = await make_recall_node(memory, history_turns=6)(state_with_history("visa?"))

    assert stateless == {"recalled": []}
    assert in_window == {"recalled": []}


async def test_memory_failures_do_not_fail_the_turn(
    store: QdrantMemoryStore, caplog: pytest.LogCaptureFixture
) -> None:
    broken = ConversationMemory(TopicEmbeddingProvider(ProviderError("secret detail")), store)

    recalled = await make_recall_node(broken, history_turns=0)(state_with_history("visa?"))
    state = state_with_history("visa?")
    state["answer"] = "An answer."
    finalized = await make_finalize_node(broken)(state)

    assert recalled == {"recalled": []}
    assert finalized["history"] == [ConversationTurn(query="visa?", answer="An answer.")]
    assert "recall failed: ProviderError" in caplog.text
    assert "write failed: ProviderError" in caplog.text
    assert "secret detail" not in caplog.text


async def test_finalize_remembers_the_turn_at_its_position(store: QdrantMemoryStore) -> None:
    memory = make_memory(store)
    state = state_with_history("How do I renew a visa?")
    state["answer"] = "At the visa office [1]."

    await make_finalize_node(memory)(state)
    stateless = state_with_history("Another visa question?", thread_id=None)
    stateless["answer"] = "Not stored."
    await make_finalize_node(memory)(stateless)

    recalled = await memory.recall("t1", "visa", before_turn=4)
    assert recalled == [
        ConversationTurn(query="How do I renew a visa?", answer="At the visa office [1].")
    ]


def test_prompts_show_recalled_turns_before_recent_turns() -> None:
    state = state_with_history("q")
    state["recalled"] = [TURNS[0]]

    messages = history_messages(state, 1)

    assert [message["content"] for message in messages] == [
        "What visa do I need?",
        "A student visa [1].",
        "How much is tuition?",
        "Tuition is 9000 [1].",
    ]
    assert render_history(state, 0) == "User: What visa do I need?\nAssistant: A student visa [1]."


def test_stream_node_event_reports_the_number_of_recalled_turns() -> None:
    assert node_event("recall", {"recalled": [TURNS[0]]}) == {"node": "recall", "recalled_turns": 1}


# --- graph -------------------------------------------------------------------------------


async def test_graph_recalls_an_old_turn_and_reports_the_recall_node(
    store: QdrantMemoryStore,
) -> None:
    memory = make_memory(store)
    chat = ScriptedChatProvider(agent=ANSWER, direct=["Visa reply.", "Library reply.", "Again."])
    search = DocumentSearch(
        StaticRetriever(),
        BudgetedContextAssembler(word_count),
        top_k=4,
        budget=ContextBudget(max_tokens=100, max_chunks=2),
    )
    nodes = WorkflowNodes(
        agent=make_agent_node(chat, ToolSet(), max_steps=2, history_turns=1),
        act=make_act_node(search, ToolSet()),
        answer=make_answer_node(chat, history_turns=1),
        verify=make_verify_node(None),
        retry=retry_node,
        finalize=make_finalize_node(memory),
        recall=make_recall_node(memory, history_turns=1),
    )
    graph = build_graph(
        nodes, checkpointer=InMemorySaver(serde=create_checkpoint_serializer())
    )
    config = {**thread_config("t1"), "recursion_limit": recursion_limit(2, 1)}

    for question in ("Which visa do I need?", "Is the library open?"):
        await graph.ainvoke(new_turn_input(question, thread_id="t1"), config)
    result = await graph.ainvoke(new_turn_input("Tell me about the visa", thread_id="t1"), config)

    assert result["node_trace"] == ["recall", "agent", "answer", "finalize"]
    assert result["recalled"] == [
        ConversationTurn(query="Which visa do I need?", answer="Visa reply.")
    ]
    agent_prompt = chat.calls_for("agent")[2][1]["content"]
    assert "User: Which visa do I need?\nAssistant: Visa reply." in agent_prompt
    assert [message["content"] for message in chat.calls_for("direct")[2][1:]] == [
        "Which visa do I need?",
        "Visa reply.",
        "Is the library open?",
        "Library reply.",
        "Tell me about the visa",
    ]
