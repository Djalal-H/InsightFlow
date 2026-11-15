"""Tests for the Langfuse trace sink with an in-memory span exporter (no network)."""

import json
import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from insightflow.api.dependencies import create_trace_sink
from insightflow.core.config import Settings
from insightflow.core.observability import (
    NodeRun,
    NullTraceSink,
    ProviderCall,
    RunRecorder,
    RunTrace,
    StoreCall,
    ToolCall,
)
from insightflow.observability.langfuse_sink import LangfuseTraceSink

START = datetime(2026, 11, 14, 12, 0, tzinfo=UTC)


def at(ms: float) -> datetime:
    return START + timedelta(milliseconds=ms)


def sample_trace(*, capture_content: bool) -> RunTrace:
    recorder = RunRecorder("req-1", thread_id="thread-1", capture_content=capture_content)
    recorder.started_at = START
    recorder.node_runs = [
        NodeRun("agent", 1, at(1), 10.0, "ok"),
        NodeRun("act", 1, at(12), 20.0, "ok"),
        NodeRun("answer", 1, at(33), 30.0, "provider_timeout"),
    ]
    recorder.provider_calls = [
        ProviderCall(
            "chat",
            "complete",
            "nim/chat",
            "agent",
            at(2),
            8.0,
            prompt_tokens=100,
            completion_tokens=20,
            input=[{"role": "user", "content": "secret prompt"}] if capture_content else None,
            output="secret output" if capture_content else None,
        ),
        ProviderCall("embedding", "embed_query", "nim/embed", "act", at(13), 4.0),
        ProviderCall("chat", "stream", "nim/chat", "answer", at(34), 29.0, "provider_timeout"),
    ]
    recorder.store_calls = [StoreCall("search", "act", at(18), 5.0, 6, "ok")]
    recorder.tool_calls = [
        ToolCall("search_documents", "insightflow", "act", at(12.5), 19.0, "ok", None)
    ]
    recorder.errors["provider_timeout"] = 1
    return RunTrace(
        name="query",
        recorder=recorder,
        ended_at=at(70),
        input="secret question" if capture_content else None,
        output=None,
        metadata={"attempts": 1, "verification": None},
    )


@pytest.fixture
def exporter() -> Iterator[InMemorySpanExporter]:
    yield InMemorySpanExporter()


def make_sink(exporter: InMemorySpanExporter, **kwargs: object) -> LangfuseTraceSink:
    # The SDK keeps one exporter per public key and process, so each test uses its own key.
    return LangfuseTraceSink(
        public_key=f"pk-lf-{uuid.uuid4().hex}",
        secret_key="sk-lf-test",
        host="http://langfuse.invalid",
        span_exporter=exporter,
        **kwargs,  # type: ignore[arg-type]
    )


def spans_by_name(exporter: InMemorySpanExporter) -> dict[str, ReadableSpan]:
    return {span.name: span for span in exporter.get_finished_spans()}


def test_a_run_becomes_one_trace_with_nodes_and_their_calls(
    exporter: InMemorySpanExporter,
) -> None:
    sink = make_sink(exporter)
    sink.export(sample_trace(capture_content=False))
    sink.flush()

    spans = spans_by_name(exporter)
    assert set(spans) == {
        "query",
        "agent",
        "act",
        "answer",
        "chat.complete",
        "embedding.embed_query",
        "chat.stream",
        "qdrant.search",
        "search_documents",
    }
    root = spans["query"]
    assert root.parent is None
    assert {span.context.trace_id for span in spans.values()} == {root.context.trace_id}
    assert root.attributes is not None
    assert root.attributes["langfuse.trace.name"] == "query"
    assert root.attributes["session.id"] == "thread-1"
    assert root.attributes["langfuse.trace.metadata.request_id"] == '"req-1"'
    assert "langfuse.trace.metadata.verification" not in root.attributes
    assert root.attributes["langfuse.observation.level"] == "ERROR"
    assert root.start_time == int(START.timestamp() * 1e9)
    assert root.end_time == int(at(70).timestamp() * 1e9)

    def parent_name(name: str) -> str:
        parent = spans[name].parent
        assert parent is not None
        return next(s.name for s in spans.values() if s.context.span_id == parent.span_id)

    assert parent_name("agent") == "query"
    assert parent_name("chat.complete") == "agent"
    assert parent_name("embedding.embed_query") == "act"
    assert parent_name("qdrant.search") == "act"
    assert parent_name("search_documents") == "act"
    assert parent_name("chat.stream") == "answer"

    generation = spans["chat.complete"].attributes
    assert generation is not None
    assert generation["langfuse.observation.type"] == "generation"
    assert generation["langfuse.observation.model.name"] == "nim/chat"
    assert json.loads(str(generation["langfuse.observation.usage_details"])) == {
        "input": 100,
        "output": 20,
    }
    assert "langfuse.observation.input" not in generation
    assert spans["embedding.embed_query"].attributes["langfuse.observation.type"] == "embedding"  # type: ignore[index]
    failed = spans["chat.stream"].attributes
    assert failed is not None
    assert failed["langfuse.observation.level"] == "ERROR"
    assert failed["langfuse.observation.status_message"] == "provider_timeout"
    assert spans["qdrant.search"].attributes["langfuse.observation.type"] == "retriever"  # type: ignore[index]
    assert spans["search_documents"].attributes["langfuse.observation.type"] == "tool"  # type: ignore[index]


def test_content_is_sent_only_with_capture(exporter: InMemorySpanExporter) -> None:
    sink = make_sink(exporter, capture_content=True)
    assert sink.capture_content
    sink.export(sample_trace(capture_content=True))
    sink.flush()

    spans = spans_by_name(exporter)
    root, generation = spans["query"].attributes, spans["chat.complete"].attributes
    assert root is not None and generation is not None
    assert root["langfuse.observation.input"] == "secret question"
    assert "secret prompt" in str(generation["langfuse.observation.input"])
    assert generation["langfuse.observation.output"] == "secret output"


def test_a_sink_without_capture_drops_recorded_content(exporter: InMemorySpanExporter) -> None:
    sink = make_sink(exporter)
    sink.export(sample_trace(capture_content=True))
    sink.flush()

    for span in exporter.get_finished_spans():
        assert span.attributes is not None
        if span.name != "query":
            assert "langfuse.observation.input" not in span.attributes
            assert "langfuse.observation.output" not in span.attributes


def test_sampling_skips_runs(exporter: InMemorySpanExporter) -> None:
    sink = make_sink(exporter, sample_rate=0.5, random_value=lambda: 0.9)
    sink.export(sample_trace(capture_content=False))
    sink.flush()
    assert exporter.get_finished_spans() == ()


def test_export_failures_are_logged_not_raised(
    exporter: InMemorySpanExporter, caplog: pytest.LogCaptureFixture
) -> None:
    sink = make_sink(exporter)
    broken = sample_trace(capture_content=False)
    broken.recorder.provider_calls.append(
        ProviderCall("unknown", "x", "m", None)  # type: ignore[arg-type]
    )
    sink.export(broken)
    sink.shutdown()
    assert "Langfuse export failed: KeyError" in caplog.text


def test_the_factory_builds_a_null_sink_unless_langfuse_is_enabled() -> None:
    # LiteLLM loads .env into the environment, so the switch is given explicitly.
    assert isinstance(
        create_trace_sink(Settings(_env_file=None, langfuse_enabled=False)), NullTraceSink
    )
    settings = Settings(
        _env_file=None,
        langfuse_enabled=True,
        langfuse_public_key=f"pk-lf-{uuid.uuid4().hex}",
        langfuse_secret_key="sk-lf-test",
        langfuse_host="http://langfuse.invalid",
        langfuse_capture_content=True,
    )
    sink = create_trace_sink(settings)
    assert isinstance(sink, LangfuseTraceSink)
    assert sink.capture_content
    sink.shutdown()


@pytest.mark.langfuse
def test_one_synthetic_trace_reaches_a_real_langfuse_server() -> None:
    host = os.environ.get("LANGFUSE_TEST_HOST")
    public_key = os.environ.get("LANGFUSE_TEST_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_TEST_SECRET_KEY")
    if not (host and public_key and secret_key):
        pytest.skip("set LANGFUSE_TEST_HOST, LANGFUSE_TEST_PUBLIC_KEY, LANGFUSE_TEST_SECRET_KEY")
    sink = LangfuseTraceSink(public_key=public_key, secret_key=secret_key, host=host)
    trace = sample_trace(capture_content=False)
    trace.recorder.started_at = datetime.now(UTC)
    sink.export(trace)
    sink.shutdown()


def test_a_second_sink_with_the_same_key_uses_the_first_exporter(
    exporter: InMemorySpanExporter,
) -> None:
    key = f"pk-lf-{uuid.uuid4().hex}"
    first = LangfuseTraceSink(
        public_key=key, secret_key="s", host="http://langfuse.invalid", span_exporter=exporter
    )
    second = LangfuseTraceSink(public_key=key, secret_key="s", host="http://langfuse.invalid")
    second.export(sample_trace(capture_content=False))
    first.flush()
    assert "query" in spans_by_name(exporter)
