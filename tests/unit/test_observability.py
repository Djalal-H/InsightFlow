"""Tests for run records, provider instrumentation, and usage parsing."""

import asyncio
from collections.abc import AsyncIterator, Sequence
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from insightflow.api.errors import describe_error
from insightflow.core.exceptions import (
    BudgetExceededError,
    InsightFlowError,
    ProviderConfigurationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    StorageError,
    ThreadNotFoundError,
    ToolTimeoutError,
)
from insightflow.core.observability import (
    CallBudget,
    NullTraceSink,
    RunRecorder,
    RunTrace,
    current_node,
    current_recorder,
    error_code,
    export_run,
    node_run,
    node_scope,
    record_error,
    record_tool_call,
    recording,
    report_usage,
    store_call,
)
from insightflow.providers.instrumented import (
    InstrumentedChatProvider,
    InstrumentedEmbeddingProvider,
    InstrumentedRerankProvider,
    instrument_chat,
    instrument_embedding,
    instrument_rerank,
)
from insightflow.providers.llm import ChatMessage, LiteLLMChatProvider, response_usage
from insightflow.providers.rerank import RerankResult
from tests.fakes import FakeRerankProvider

MESSAGES: list[ChatMessage] = [{"role": "user", "content": "secret question"}]


class UsageChatProvider:
    """Reports token use like the LiteLLM adapter does, or raises one error."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        del messages
        if self.error is not None:
            raise self.error
        report_usage(12, 3)
        return "secret answer"

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        del messages
        yield "one "
        if self.error is not None:
            raise self.error
        report_usage(10, None)
        yield "two"
        report_usage(None, 2)


class UsageEmbeddingProvider:
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        report_usage(len(texts))
        return [[1.0] for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        del text
        report_usage(4)
        return [1.0]

    def count_tokens(self, text: str) -> int:
        return len(text.split())


@pytest.mark.parametrize(
    "exc",
    [
        ProviderConfigurationError(),
        ProviderTimeoutError(),
        ProviderRateLimitError(),
        ProviderError(),
        StorageError(),
        ThreadNotFoundError(),
    ],
)
def test_error_codes_match_the_public_api_codes(exc: InsightFlowError) -> None:
    _, detail = describe_error(exc)
    assert error_code(exc) == detail["code"]


def test_error_codes_for_failures_without_a_public_mapping() -> None:
    assert error_code(ToolTimeoutError()) == "tool_timeout"
    assert error_code(BudgetExceededError()) == "budget_exceeded"
    assert error_code(ValueError("secret detail")) == "internal_error"


async def test_chat_call_is_recorded_with_usage_node_and_no_content_by_default() -> None:
    provider = InstrumentedChatProvider(UsageChatProvider(), model="nim/chat")
    with recording(RunRecorder("req-1")) as recorder, node_scope("answer"):
        assert await provider.complete(MESSAGES) == "secret answer"

    [call] = recorder.provider_calls
    assert (call.kind, call.operation, call.model, call.node) == (
        "chat",
        "complete",
        "nim/chat",
        "answer",
    )
    assert (call.prompt_tokens, call.completion_tokens, call.outcome) == (12, 3, "ok")
    assert call.input is None and call.output is None
    assert call.latency_ms >= 0
    totals = recorder.totals("chat")
    assert (totals.calls, totals.prompt_tokens, totals.completion_tokens) == (1, 12, 3)
    assert totals.unreported_usage == 0


async def test_chat_content_is_kept_only_when_the_recorder_captures_it() -> None:
    provider = InstrumentedChatProvider(UsageChatProvider(), model="m")
    with recording(RunRecorder(capture_content=True)) as recorder:
        await provider.complete(MESSAGES)

    [call] = recorder.provider_calls
    assert call.input == MESSAGES
    assert call.output == "secret answer"


async def test_failed_chat_call_records_its_error_code_and_reraises() -> None:
    provider = InstrumentedChatProvider(UsageChatProvider(ProviderTimeoutError()), model="m")
    with recording(RunRecorder()) as recorder, pytest.raises(ProviderTimeoutError):
        await provider.complete(MESSAGES)

    [call] = recorder.provider_calls
    assert call.outcome == "provider_timeout"
    assert recorder.totals("chat").errors == 1


async def test_streamed_chat_sums_usage_reported_between_pieces() -> None:
    provider = InstrumentedChatProvider(UsageChatProvider(), model="m")
    with recording(RunRecorder(capture_content=True)) as recorder:
        pieces = [piece async for piece in provider.stream(MESSAGES)]

    assert pieces == ["one ", "two"]
    [call] = recorder.provider_calls
    assert (call.operation, call.prompt_tokens, call.completion_tokens) == ("stream", 10, 2)
    assert call.output == "one two"


async def test_streamed_chat_failure_is_recorded() -> None:
    provider = InstrumentedChatProvider(UsageChatProvider(ProviderError()), model="m")
    with recording(RunRecorder()) as recorder, pytest.raises(ProviderError):
        [piece async for piece in provider.stream(MESSAGES)]

    assert recorder.provider_calls[0].outcome == "provider_error"


async def test_a_stream_closed_early_is_recorded_as_cancelled() -> None:
    provider = InstrumentedChatProvider(UsageChatProvider(), model="m")
    with recording(RunRecorder()) as recorder:
        stream = provider.stream(MESSAGES)
        assert await anext(stream) == "one "
        await stream.aclose()  # type: ignore[attr-defined]

    [call] = recorder.provider_calls
    assert call.outcome == "cancelled"
    assert recorder.totals("chat").errors == 0


async def test_usage_reported_outside_a_call_is_ignored() -> None:
    report_usage(5, 5)  # no active call: nothing happens
    with recording(RunRecorder()) as recorder:
        report_usage(5, 5)
    assert recorder.provider_calls == []


async def test_calls_without_a_recorder_are_not_recorded() -> None:
    provider = InstrumentedChatProvider(UsageChatProvider(), model="m")
    assert current_recorder() is None
    assert await provider.complete(MESSAGES) == "secret answer"


async def test_embedding_and_rerank_calls_are_recorded() -> None:
    embedding = InstrumentedEmbeddingProvider(UsageEmbeddingProvider(), model="nim/embed")
    rerank = InstrumentedRerankProvider(FakeRerankProvider("b"), model="nim/rerank")
    with recording(RunRecorder()) as recorder:
        assert await embedding.embed_documents(["a", "b"]) == [[1.0], [1.0]]
        assert await embedding.embed_query("q") == [1.0]
        assert embedding.count_tokens("two words") == 2
        results = await rerank.rerank("q", ["a", "b"], top_n=1)

    assert results == [RerankResult(index=1, score=1.0)]
    assert [(c.kind, c.operation, c.prompt_tokens) for c in recorder.provider_calls] == [
        ("embedding", "embed_documents", 2),
        ("embedding", "embed_query", 4),
        ("rerank", "rerank", None),
    ]
    assert recorder.totals("embedding").prompt_tokens == 6
    assert recorder.totals("rerank").unreported_usage == 1


async def test_failed_embedding_and_rerank_calls_are_recorded() -> None:
    failing = AsyncMock(side_effect=ProviderRateLimitError())
    inner = SimpleNamespace(embed_documents=failing, embed_query=failing, rerank=failing)
    embedding = InstrumentedEmbeddingProvider(inner, model="e")  # type: ignore[arg-type]
    rerank = InstrumentedRerankProvider(inner, model="r")  # type: ignore[arg-type]
    with recording(RunRecorder()) as recorder:
        for call in (
            embedding.embed_documents(["a"]),
            embedding.embed_query("q"),
            rerank.rerank("q", ["a"], top_n=1),
        ):
            with pytest.raises(ProviderRateLimitError):
                await call

    assert [c.outcome for c in recorder.provider_calls] == ["provider_rate_limited"] * 3


def test_instrumenting_twice_wraps_once() -> None:
    chat = instrument_chat(UsageChatProvider(), model="m")
    embedding = instrument_embedding(UsageEmbeddingProvider(), model="e")
    rerank = instrument_rerank(FakeRerankProvider(), model="r")
    assert instrument_chat(chat, model="m") is chat
    assert instrument_embedding(embedding, model="e") is embedding
    assert instrument_rerank(rerank, model="r") is rerank


async def test_budget_stops_chat_calls_and_token_use() -> None:
    provider = InstrumentedChatProvider(UsageChatProvider(), model="m")
    with recording(RunRecorder(budget=CallBudget(max_chat_calls=1))):
        await provider.complete(MESSAGES)
        with pytest.raises(BudgetExceededError):
            await provider.complete(MESSAGES)

    with recording(RunRecorder(budget=CallBudget(max_tokens=10))) as recorder:
        await provider.complete(MESSAGES)  # 15 tokens: over the budget after this call
        with pytest.raises(BudgetExceededError):
            await provider.complete(MESSAGES)
    assert len(recorder.provider_calls) == 1


async def test_store_calls_record_results_and_failures() -> None:
    with recording(RunRecorder()) as recorder, node_scope("act"):
        async with store_call("search") as scope:
            scope.results = 3
        with pytest.raises(StorageError):
            async with store_call("get_chunks"):
                raise StorageError("secret")

    first, second = recorder.store_calls
    assert (first.operation, first.results, first.outcome, first.node) == ("search", 3, "ok", "act")
    assert (second.operation, second.results, second.outcome) == (
        "get_chunks",
        0,
        "storage_unavailable",
    )


async def test_node_runs_set_the_node_and_record_failures() -> None:
    with recording(RunRecorder()) as recorder:
        async with node_run("agent", 1):
            assert current_node() == "agent"
        with pytest.raises(ProviderError):
            async with node_run("answer", 2):
                raise ProviderError
    assert current_node() is None
    assert [(r.name, r.attempt, r.outcome) for r in recorder.node_runs] == [
        ("agent", 1, "ok"),
        ("answer", 2, "provider_error"),
    ]
    assert set(recorder.node_timings_ms()) == {"agent", "answer"}


async def test_concurrent_runs_keep_their_own_records() -> None:
    provider = InstrumentedChatProvider(UsageChatProvider(), model="m")

    async def run(name: str, calls: int) -> RunRecorder:
        with recording(RunRecorder(name)) as recorder:

            async def node() -> None:  # LangGraph runs nodes as tasks with a copied context
                async with node_run(name, 1):
                    for _ in range(calls):
                        await provider.complete(MESSAGES)
                        await asyncio.sleep(0)

            await asyncio.create_task(node())
        return recorder

    first, second = await asyncio.gather(run("a", 1), run("b", 3))
    assert len(first.provider_calls) == 1 and {c.node for c in first.provider_calls} == {"a"}
    assert len(second.provider_calls) == 3 and {c.node for c in second.provider_calls} == {"b"}


async def test_tool_calls_and_errors_are_recorded() -> None:
    from datetime import UTC, datetime

    with recording(RunRecorder()) as recorder:
        record_tool_call(
            tool="calculate",
            server="calculator",
            started_at=datetime.now(UTC),
            duration_ms=1.234,
            status="error",
            error="timeout",
        )
        record_error("provider_timeout")
        record_error("provider_timeout")
    record_error("ignored")  # no recorder

    [tool] = recorder.tool_calls
    assert (tool.tool, tool.duration_ms, tool.error) == ("calculate", 1.2, "timeout")
    assert recorder.summary()["errors"] == {"provider_timeout": 2}
    assert recorder.summary()["tool_calls"] == 1


def test_summary_lists_only_used_provider_kinds() -> None:
    recorder = RunRecorder()
    assert "chat_calls" not in recorder.summary()


def test_export_run_drops_content_without_capture_and_survives_a_failing_sink() -> None:
    received: list[RunTrace] = []

    class Sink(NullTraceSink):
        def export(self, trace: RunTrace) -> None:
            received.append(trace)

    class BrokenSink(NullTraceSink):
        def export(self, trace: RunTrace) -> None:
            raise RuntimeError("down")

    export_run(Sink(), RunRecorder(), name="query", input="q", output="a")
    export_run(Sink(), RunRecorder(capture_content=True), name="query", input="q", output="a")
    export_run(BrokenSink(), RunRecorder(), name="query")
    NullTraceSink().export(received[0])
    NullTraceSink().flush()

    assert [(t.input, t.output) for t in received] == [(None, None), ("q", "a")]


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (SimpleNamespace(usage=SimpleNamespace(prompt_tokens=7, completion_tokens=2)), (7, 2)),
        ({"usage": {"prompt_tokens": 5}}, (5, None)),
        (SimpleNamespace(usage=None), (None, None)),
        (SimpleNamespace(), (None, None)),
        (SimpleNamespace(usage={"prompt_tokens": True, "completion_tokens": -1}), (None, None)),
    ],
)
def test_response_usage_reads_only_valid_counts(
    response: object, expected: tuple[int | None, int | None]
) -> None:
    assert response_usage(response) == expected


async def test_litellm_chat_adapter_reports_usage_to_the_instrumented_call() -> None:
    with patch(
        "insightflow.providers.llm.get_llm_provider",
        return_value=("test-model", "openai", None, None),
    ):
        adapter = LiteLLMChatProvider(model="openai/test-model")
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))],
        usage=SimpleNamespace(prompt_tokens=9, completion_tokens=1),
    )
    provider = InstrumentedChatProvider(adapter, model="openai/test-model")
    with (
        patch("insightflow.providers.llm.acompletion", AsyncMock(return_value=response)),
        recording(RunRecorder()) as recorder,
    ):
        assert await provider.complete(MESSAGES) == "hi"

    [call] = recorder.provider_calls
    assert (call.prompt_tokens, call.completion_tokens) == (9, 1)
