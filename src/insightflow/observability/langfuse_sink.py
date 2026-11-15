"""Langfuse trace sink: one trace per finished run, sent with the Langfuse SDK.

This is the only module that imports `langfuse` or OpenTelemetry. The run is exported after it
ends, from the records in `RunRecorder`, so spans get their real start and end times and no
tracing context has to cross the workflow. The spans use Langfuse's OpenTelemetry attribute
names (`langfuse._client.attributes.LangfuseOtelSpanAttributes`) on a private tracer provider;
the global OpenTelemetry provider is not changed.

Trace layout: the request is the root span (trace name, `session.id` = thread), each workflow
node is a child span, and provider, store, and tool calls are children of the node they ran in.
Model inputs and outputs are attached only when content capture is on.
"""

import json
import logging
import random
from collections.abc import Callable
from datetime import datetime
from typing import Any

from langfuse import Langfuse
from langfuse._client.attributes import LangfuseOtelSpanAttributes as Attr
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter

from insightflow.core.observability import (
    OK,
    NodeRun,
    ProviderCall,
    RunTrace,
    StoreCall,
    ToolCall,
)

logger = logging.getLogger(__name__)

TRACER_NAME = "insightflow"
_OBSERVATION_TYPES = {"chat": "generation", "embedding": "embedding", "rerank": "generation"}


def _ns(moment: datetime) -> int:
    return int(moment.timestamp() * 1_000_000_000)


def _end_ns(moment: datetime, latency_ms: float) -> int:
    return _ns(moment) + int(latency_ms * 1_000_000)


def _json(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=False)


def _drop_none(attributes: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in attributes.items() if value is not None}


def _level(outcome: str) -> dict[str, Any]:
    if outcome == OK:
        return {}
    return {Attr.OBSERVATION_LEVEL: "ERROR", Attr.OBSERVATION_STATUS_MESSAGE: outcome}


class LangfuseTraceSink:
    """Send finished runs to Langfuse. Failures are logged and the run is dropped."""

    def __init__(
        self,
        *,
        public_key: str,
        secret_key: str,
        host: str,
        capture_content: bool = False,
        sample_rate: float = 1.0,
        span_exporter: SpanExporter | None = None,
        random_value: Callable[[], float] = random.random,
    ) -> None:
        self._capture_content = capture_content
        self._sample_rate = sample_rate
        self._random = random_value
        self._provider = TracerProvider()
        # Every span of this private provider is ours, so the SDK's scope filter is not needed.
        self._client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            base_url=host,
            tracer_provider=self._provider,
            should_export_span=lambda span: True,
            span_exporter=span_exporter,
        )
        # The SDK keeps one set of resources per public key and process: a second client with
        # the same key reuses the first provider and exporter, so spans must go there.
        resources = getattr(self._client, "_resources", None)
        provider = getattr(resources, "tracer_provider", None)
        if isinstance(provider, TracerProvider):
            self._provider = provider
        self._tracer = self._provider.get_tracer(TRACER_NAME)

    @property
    def capture_content(self) -> bool:
        return self._capture_content

    def export(self, trace: RunTrace) -> None:
        if self._sample_rate < 1.0 and self._random() >= self._sample_rate:
            return
        try:
            self._export(trace)
        except Exception as exc:
            logger.warning("Langfuse export failed: %s", type(exc).__name__)

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception as exc:
            logger.warning("Langfuse flush failed: %s", type(exc).__name__)

    def shutdown(self) -> None:
        try:
            self._client.shutdown()
        except Exception as exc:
            logger.warning("Langfuse shutdown failed: %s", type(exc).__name__)

    def _export(self, trace: RunTrace) -> None:
        recorder = trace.recorder
        root_attributes = _drop_none(
            {
                Attr.TRACE_NAME: trace.name,
                Attr.TRACE_SESSION_ID: recorder.thread_id,
                Attr.OBSERVATION_TYPE: "span",
                Attr.OBSERVATION_INPUT: trace.input,
                Attr.OBSERVATION_OUTPUT: trace.output,
                f"{Attr.TRACE_METADATA}.request_id": _json(recorder.request_id),
                **{
                    f"{Attr.TRACE_METADATA}.{key}": _json(value)
                    for key, value in trace.metadata.items()
                    if value is not None
                },
                **({Attr.OBSERVATION_LEVEL: "ERROR"} if recorder.errors else {}),
            }
        )
        root = self._tracer.start_span(
            trace.name, start_time=_ns(recorder.started_at), attributes=root_attributes
        )
        root_context = otel_trace.set_span_in_context(root)
        node_contexts: list[tuple[NodeRun, Any]] = []
        for run in recorder.node_runs:
            span = self._tracer.start_span(
                run.name,
                context=root_context,
                start_time=_ns(run.started_at),
                attributes={
                    Attr.OBSERVATION_TYPE: "span",
                    f"{Attr.OBSERVATION_METADATA}.attempt": _json(run.attempt),
                    **_level(run.outcome),
                },
            )
            node_contexts.append((run, otel_trace.set_span_in_context(span)))
            span.end(end_time=_end_ns(run.started_at, run.latency_ms))

        def parent(node: str | None, started_at: datetime) -> Any:
            moment = _ns(started_at)
            for run, context in node_contexts:
                start = _ns(run.started_at)
                if run.name == node and start <= moment <= _end_ns(run.started_at, run.latency_ms):
                    return context
            return root_context

        for call in recorder.provider_calls:
            self._provider_span(call, parent(call.node, call.started_at))
        for store in recorder.store_calls:
            self._store_span(store, parent(store.node, store.started_at))
        for tool in recorder.tool_calls:
            self._tool_span(tool, parent(tool.node, tool.started_at))
        root.end(end_time=_ns(trace.ended_at))

    def _provider_span(self, call: ProviderCall, context: Any) -> None:
        usage = _drop_none({"input": call.prompt_tokens, "output": call.completion_tokens})
        attributes = _drop_none(
            {
                Attr.OBSERVATION_TYPE: _OBSERVATION_TYPES[call.kind],
                Attr.OBSERVATION_MODEL: call.model or None,
                Attr.OBSERVATION_USAGE_DETAILS: _json(usage) if usage else None,
                Attr.OBSERVATION_INPUT: (
                    _json(call.input)
                    if self._capture_content and call.input is not None
                    else None
                ),
                Attr.OBSERVATION_OUTPUT: call.output if self._capture_content else None,
                f"{Attr.OBSERVATION_METADATA}.operation": _json(call.operation),
                **_level(call.outcome),
            }
        )
        span = self._tracer.start_span(
            f"{call.kind}.{call.operation}",
            context=context,
            start_time=_ns(call.started_at),
            attributes=attributes,
        )
        span.end(end_time=_end_ns(call.started_at, call.latency_ms))

    def _store_span(self, call: StoreCall, context: Any) -> None:
        span = self._tracer.start_span(
            f"qdrant.{call.operation}",
            context=context,
            start_time=_ns(call.started_at),
            attributes={
                Attr.OBSERVATION_TYPE: "retriever",
                f"{Attr.OBSERVATION_METADATA}.results": _json(call.results),
                **_level(call.outcome),
            },
        )
        span.end(end_time=_end_ns(call.started_at, call.latency_ms))

    def _tool_span(self, call: ToolCall, context: Any) -> None:
        span = self._tracer.start_span(
            call.tool,
            context=context,
            start_time=_ns(call.started_at),
            attributes={
                Attr.OBSERVATION_TYPE: "tool",
                f"{Attr.OBSERVATION_METADATA}.server": _json(call.server),
                **_level(call.error or OK),
            },
        )
        span.end(end_time=_end_ns(call.started_at, call.duration_ms))
