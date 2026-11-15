"""Request IDs, log lines, usage, and trace export through the HTTP API."""

import logging
from typing import Any

import pytest

from insightflow.api.dependencies import get_trace_sink
from insightflow.core.exceptions import ProviderTimeoutError, StorageError
from insightflow.core.observability import NullTraceSink, RunTrace
from tests.fakes import ScriptedChatProvider, StaticRetriever
from tests.integration.test_query_api import client_for, make_app
from tests.integration.test_query_stream_api import stream_query


class RecordingSink(NullTraceSink):
    def __init__(self, *, capture_content: bool = False, broken: bool = False) -> None:
        self.capture_content = capture_content
        self.broken = broken
        self.traces: list[RunTrace] = []

    def export(self, trace: RunTrace) -> None:
        if self.broken:
            raise RuntimeError("tracing service down")
        self.traces.append(trace)


def app_with_sink(sink: RecordingSink, **kwargs: Any) -> Any:
    application = make_app(ScriptedChatProvider(), **kwargs)
    application.dependency_overrides[get_trace_sink] = lambda: sink
    return application


async def test_a_valid_request_id_is_kept_and_returned() -> None:
    async with client_for(make_app(ScriptedChatProvider())) as client:
        response = await client.post(
            "/query", json={"query": "Q"}, headers={"X-Request-ID": "client-42"}
        )
    assert response.headers["x-request-id"] == "client-42"


@pytest.mark.parametrize("header", ["bad id with spaces", "x" * 129, "<script>"])
async def test_an_unsafe_request_id_is_replaced(header: str) -> None:
    async with client_for(make_app(ScriptedChatProvider())) as client:
        response = await client.post(
            "/query", json={"query": "Q"}, headers={"X-Request-ID": header}
        )
    assert response.headers["x-request-id"] != header
    assert len(response.headers["x-request-id"]) == 32


async def test_health_requests_get_a_request_id() -> None:
    async with client_for(make_app(ScriptedChatProvider())) as client:
        response = await client.get("/health/live")
    assert len(response.headers["x-request-id"]) == 32


async def test_one_request_id_joins_the_access_summary_and_node_lines(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from insightflow.core.logging import context_fields

    seen: list[dict[str, Any]] = []

    class ContextHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            seen.append({"logger": record.name, "message": record.getMessage(), **context_fields()})

    handler = ContextHandler(level=logging.DEBUG)
    logging.getLogger().addHandler(handler)
    caplog.set_level(logging.DEBUG)
    try:
        async with client_for(make_app(ScriptedChatProvider())) as client:
            response = await client.post(
                "/query", json={"query": "Q"}, headers={"X-Request-ID": "join-1"}
            )
    finally:
        logging.getLogger().removeHandler(handler)

    assert response.status_code == 200
    ours = [entry for entry in seen if entry.get("request_id") == "join-1"]
    messages = {entry["message"] for entry in ours}
    assert {"request", "query completed", "node run", "provider call", "tool call"} <= messages
    assert {entry["node"] for entry in ours if entry["message"] == "provider call"} == {
        "agent",
        "answer",
    }
    summary = next(r for r in caplog.records if r.getMessage() == "query completed")
    fields = summary.fields  # type: ignore[attr-defined]
    assert fields["outcome"] == "ok"
    assert fields["chat_calls"] == 3
    assert fields["verification"] == "passed"
    assert "Q" not in str(fields)  # the question is not logged


async def test_a_failed_request_logs_and_traces_its_error_code(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sink = RecordingSink()
    application = make_app(ScriptedChatProvider(answer=ProviderTimeoutError("secret detail")))
    application.dependency_overrides[get_trace_sink] = lambda: sink
    caplog.set_level(logging.INFO)

    async with client_for(application) as client:
        response = await client.post("/query", json={"query": "Q"})

    assert response.status_code == 504
    failed = next(r for r in caplog.records if r.getMessage() == "request failed")
    assert failed.fields == {  # type: ignore[attr-defined]
        "code": "provider_timeout",
        "error_type": "ProviderTimeoutError",
    }
    assert "secret detail" not in caplog.text
    [trace] = sink.traces
    assert trace.recorder.errors == {"provider_timeout": 1}
    assert trace.output is None
    summary = next(r for r in caplog.records if r.getMessage() == "query completed")
    assert summary.fields["outcome"] == "error"  # type: ignore[attr-defined]


async def test_the_trace_sink_receives_the_run_without_content_by_default() -> None:
    sink = RecordingSink()
    async with client_for(app_with_sink(sink)) as client:
        response = await client.post("/query", json={"query": "secret question"})

    assert response.status_code == 200
    [trace] = sink.traces
    assert trace.name == "query"
    assert trace.recorder.request_id == response.headers["x-request-id"]
    assert [run.name for run in trace.recorder.node_runs] == response.json()["metadata"]["nodes"]
    assert (trace.input, trace.output) == (None, None)
    assert all(call.input is None for call in trace.recorder.provider_calls)
    assert trace.metadata == {"attempts": 1, "verification": "passed"}


async def test_content_reaches_the_sink_only_when_it_asks_for_it() -> None:
    sink = RecordingSink(capture_content=True)
    async with client_for(app_with_sink(sink)) as client:
        await client.post("/query", json={"query": "secret question"})

    [trace] = sink.traces
    assert trace.input == "secret question"
    assert trace.output == "Grounded answer [1]."
    assert all(call.input is not None for call in trace.recorder.provider_calls)


async def test_a_broken_sink_does_not_fail_the_request() -> None:
    async with client_for(app_with_sink(RecordingSink(broken=True))) as client:
        response = await client.post("/query", json={"query": "Q"})
    assert response.status_code == 200


async def test_the_stream_final_event_has_usage_and_the_run_is_traced() -> None:
    sink = RecordingSink()
    status, _, events = await stream_query(app_with_sink(sink), {"query": "Q"})

    assert status == 200
    final = dict(events)["final"]
    assert final["metadata"]["usage"]["chat"]["calls"] == 3
    assert final["metadata"]["usage"]["retrieval_calls"] == 1
    [trace] = sink.traces
    assert trace.name == "query_stream"
    operations = [call.operation for call in trace.recorder.provider_calls]
    assert operations == ["complete", "complete", "stream"]


async def test_a_stream_error_is_recorded_and_traced() -> None:
    sink = RecordingSink()
    application = app_with_sink(sink, retriever=StaticRetriever(error=StorageError("secret")))
    status, _, events = await stream_query(application, {"query": "Q"})

    assert status == 200
    assert dict(events)["error"]["error"]["code"] == "storage_unavailable"
    [trace] = sink.traces
    assert trace.recorder.errors == {"storage_unavailable": 1}
