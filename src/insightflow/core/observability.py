"""Per-request run context and records of provider, store, tool, and node calls.

A `RunRecorder` collects what one run did. It is found through a context variable, so the
adapters and nodes that report to it need no extra arguments; without an active recorder every
report is ignored. LangGraph copies the context into its node tasks, so all nodes of a request
report to the recorder of that request.

The records feed three outputs: log lines (`insightflow.trace` logger), the usage totals in the
API response, and an optional `TraceSink` (for example Langfuse) that receives the finished run.
No provider, storage, or tracing library type is stored here.
"""

import logging
import time
import uuid
from collections import Counter
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from insightflow.core.exceptions import (
    BudgetExceededError,
    ProviderConfigurationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    StorageError,
    ThreadNotFoundError,
    ToolError,
    ToolTimeoutError,
)

logger = logging.getLogger("insightflow.trace")

ProviderKind = Literal["chat", "embedding", "rerank"]
PROVIDER_KINDS: tuple[ProviderKind, ...] = ("chat", "embedding", "rerank")
OK = "ok"
CANCELLED = "cancelled"


def error_code(exc: BaseException) -> str:
    """The stable category of a failure, as the API reports it; never the message."""
    for error_type, code in _ERROR_CODES:
        if isinstance(exc, error_type):
            return code
    return "internal_error"


_ERROR_CODES: tuple[tuple[type[BaseException], str], ...] = (
    (ProviderConfigurationError, "provider_configuration_error"),
    (ProviderTimeoutError, "provider_timeout"),
    (ProviderRateLimitError, "provider_rate_limited"),
    (ProviderError, "provider_error"),
    (StorageError, "storage_unavailable"),
    (ThreadNotFoundError, "thread_not_found"),
    (ToolTimeoutError, "tool_timeout"),
    (ToolError, "tool_unavailable"),
    (BudgetExceededError, "budget_exceeded"),
)


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass
class ProviderCall:
    """One hosted-model request. Tokens are `None` when the provider did not report them.

    `input` and `output` are kept only when the recorder captures content.
    """

    kind: ProviderKind
    operation: str
    model: str
    node: str | None
    started_at: datetime = field(default_factory=_now)
    latency_ms: float = 0.0
    outcome: str = OK
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    input: Any = None
    output: str | None = None


@dataclass(frozen=True)
class StoreCall:
    """One vector-store operation and how many records it returned or wrote."""

    operation: str
    node: str | None
    started_at: datetime
    latency_ms: float
    results: int
    outcome: str


@dataclass(frozen=True)
class ToolCall:
    """One agent action run by the `act` node (local document search or an MCP tool)."""

    tool: str
    server: str
    node: str | None
    started_at: datetime
    duration_ms: float
    status: str
    error: str | None


@dataclass(frozen=True)
class NodeRun:
    """One workflow node execution."""

    name: str
    attempt: int
    started_at: datetime
    latency_ms: float
    outcome: str


@dataclass(frozen=True)
class CallBudget:
    """Limits for a run; `None` means no limit. Used by evaluation runs."""

    max_chat_calls: int | None = None
    max_tokens: int | None = None


@dataclass(frozen=True)
class ProviderTotals:
    """Sums over the calls of one provider kind."""

    calls: int
    errors: int
    prompt_tokens: int | None
    completion_tokens: int | None
    unreported_usage: int


class RunRecorder:
    """Everything that one request or evaluation case did, in order."""

    def __init__(
        self,
        request_id: str | None = None,
        *,
        thread_id: str | None = None,
        capture_content: bool = False,
        budget: CallBudget | None = None,
    ) -> None:
        self.request_id = request_id or uuid.uuid4().hex
        self.thread_id = thread_id
        self.capture_content = capture_content
        self.budget = budget or CallBudget()
        self.started_at = _now()
        self.provider_calls: list[ProviderCall] = []
        self.store_calls: list[StoreCall] = []
        self.tool_calls: list[ToolCall] = []
        self.node_runs: list[NodeRun] = []
        self.errors: Counter[str] = Counter()

    def totals(self, kind: ProviderKind) -> ProviderTotals:
        calls = [call for call in self.provider_calls if call.kind == kind]
        reported = [call for call in calls if call.prompt_tokens is not None]
        completion = [c.completion_tokens for c in reported if c.completion_tokens is not None]
        return ProviderTotals(
            calls=len(calls),
            errors=sum(1 for call in calls if call.outcome not in (OK, CANCELLED)),
            prompt_tokens=sum(c.prompt_tokens or 0 for c in reported) if reported else None,
            completion_tokens=sum(completion) if completion else None,
            unreported_usage=len(calls) - len(reported),
        )

    def total_tokens(self) -> int:
        return sum(
            (call.prompt_tokens or 0) + (call.completion_tokens or 0)
            for call in self.provider_calls
        )

    def node_timings_ms(self) -> dict[str, float]:
        timings: dict[str, float] = {}
        for run in self.node_runs:
            timings[run.name] = round(timings.get(run.name, 0.0) + run.latency_ms, 1)
        return timings

    def check_budget(self, kind: ProviderKind) -> None:
        """Raise `BudgetExceededError` before a call that the budget does not allow."""
        budget = self.budget
        if budget.max_chat_calls is not None and kind == "chat":
            if self.totals("chat").calls >= budget.max_chat_calls:
                raise BudgetExceededError("chat call budget reached")
        if budget.max_tokens is not None and self.total_tokens() >= budget.max_tokens:
            raise BudgetExceededError("token budget reached")

    def summary(self) -> dict[str, Any]:
        """Flat fields for the per-request log line."""
        fields: dict[str, Any] = {}
        for kind in PROVIDER_KINDS:
            totals = self.totals(kind)
            if totals.calls:
                fields[f"{kind}_calls"] = totals.calls
                fields[f"{kind}_errors"] = totals.errors
                fields[f"{kind}_prompt_tokens"] = totals.prompt_tokens
                fields[f"{kind}_completion_tokens"] = totals.completion_tokens
        fields["store_calls"] = len(self.store_calls)
        fields["tool_calls"] = len(self.tool_calls)
        fields["nodes"] = len(self.node_runs)
        if self.errors:
            fields["errors"] = dict(self.errors)
        return fields


_recorder: ContextVar[RunRecorder | None] = ContextVar("insightflow_recorder", default=None)
_node: ContextVar[str | None] = ContextVar("insightflow_node", default=None)
_usage_target: ContextVar[ProviderCall | None] = ContextVar("insightflow_usage", default=None)


def current_recorder() -> RunRecorder | None:
    return _recorder.get()


def current_node() -> str | None:
    return _node.get()


@contextmanager
def recording(recorder: RunRecorder) -> Iterator[RunRecorder]:
    """Make `recorder` the active recorder inside the block."""
    token = _recorder.set(recorder)
    try:
        yield recorder
    finally:
        _recorder.reset(token)


@contextmanager
def node_scope(name: str) -> Iterator[None]:
    """Attribute the reports inside the block to the workflow node `name`."""
    token = _node.set(name)
    try:
        yield
    finally:
        _node.reset(token)


def record_error(code: str) -> None:
    recorder = current_recorder()
    if recorder is not None:
        recorder.errors[code] += 1


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def start_provider_call(
    kind: ProviderKind, operation: str, model: str, content: Any = None
) -> ProviderCall:
    """Check the budget and open a provider call record (finish it with `finish_provider_call`)."""
    recorder = current_recorder()
    if recorder is not None:
        recorder.check_budget(kind)
    call = ProviderCall(kind=kind, operation=operation, model=model, node=current_node())
    if recorder is not None and recorder.capture_content:
        call.input = content
    return call


def finish_provider_call(
    call: ProviderCall, started: float, *, error: BaseException | None = None
) -> None:
    call.latency_ms = _elapsed_ms(started)
    if error is not None:
        call.outcome = CANCELLED if isinstance(error, GeneratorExit) else error_code(error)
    recorder = current_recorder()
    if recorder is not None:
        if not recorder.capture_content:
            call.output = None
        recorder.provider_calls.append(call)
    level = logging.DEBUG if call.outcome in (OK, CANCELLED) else logging.WARNING
    logger.log(
        level,
        "provider call",
        extra={
            "fields": {
                "kind": call.kind,
                "operation": call.operation,
                "model": call.model,
                "latency_ms": call.latency_ms,
                "outcome": call.outcome,
                "prompt_tokens": call.prompt_tokens,
                "completion_tokens": call.completion_tokens,
            }
        },
    )


@contextmanager
def usage_target(call: ProviderCall) -> Iterator[None]:
    """Let the adapter inside the block report token use for `call`."""
    token = _usage_target.set(call)
    try:
        yield
    finally:
        _usage_target.reset(token)


def report_usage(prompt_tokens: int | None, completion_tokens: int | None = None) -> None:
    """Called by provider adapters with the token counts that the provider returned."""
    call = _usage_target.get()
    if call is None:
        return
    if prompt_tokens is not None:
        call.prompt_tokens = (call.prompt_tokens or 0) + prompt_tokens
    if completion_tokens is not None:
        call.completion_tokens = (call.completion_tokens or 0) + completion_tokens


class StoreCallScope:
    """Set `results` inside a `store_call` block to record how many records it handled."""

    results: int = 0


@asynccontextmanager
async def store_call(operation: str) -> AsyncIterator[StoreCallScope]:
    """Time one vector-store operation and record its outcome."""
    scope = StoreCallScope()
    started_at, started = _now(), time.perf_counter()
    outcome = OK
    try:
        yield scope
    except BaseException as exc:
        outcome = error_code(exc)
        raise
    finally:
        call = StoreCall(
            operation=operation,
            node=current_node(),
            started_at=started_at,
            latency_ms=_elapsed_ms(started),
            results=scope.results,
            outcome=outcome,
        )
        recorder = current_recorder()
        if recorder is not None:
            recorder.store_calls.append(call)
        logger.log(
            logging.DEBUG if outcome == OK else logging.WARNING,
            "store call",
            extra={
                "fields": {
                    "operation": operation,
                    "latency_ms": call.latency_ms,
                    "results": call.results,
                    "outcome": outcome,
                }
            },
        )


def record_tool_call(
    *,
    tool: str,
    server: str,
    started_at: datetime,
    duration_ms: float,
    status: str,
    error: str | None,
) -> None:
    call = ToolCall(
        tool=tool,
        server=server,
        node=current_node(),
        started_at=started_at,
        duration_ms=round(duration_ms, 1),
        status=status,
        error=error,
    )
    recorder = current_recorder()
    if recorder is not None:
        recorder.tool_calls.append(call)
    logger.log(
        logging.DEBUG if status == OK else logging.WARNING,
        "tool call",
        extra={
            "fields": {
                "tool": tool,
                "server": server,
                "duration_ms": call.duration_ms,
                "status": status,
                "error": error,
            }
        },
    )


@asynccontextmanager
async def node_run(name: str, attempt: int) -> AsyncIterator[None]:
    """Run a workflow node in its scope and record its latency and outcome."""
    started_at, started = _now(), time.perf_counter()
    outcome = OK
    with node_scope(name):
        try:
            yield
        except BaseException as exc:
            outcome = error_code(exc)
            raise
        finally:
            run = NodeRun(
                name=name,
                attempt=attempt,
                started_at=started_at,
                latency_ms=_elapsed_ms(started),
                outcome=outcome,
            )
            recorder = current_recorder()
            if recorder is not None:
                recorder.node_runs.append(run)
            logger.debug(
                "node run",
                extra={
                    "fields": {
                        "attempt": attempt,
                        "latency_ms": run.latency_ms,
                        "outcome": outcome,
                    }
                },
            )


@dataclass(frozen=True)
class RunTrace:
    """A finished run as a trace sink receives it. `input`/`output` only with content capture."""

    name: str
    recorder: RunRecorder
    ended_at: datetime
    input: str | None = None
    output: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class TraceSink(Protocol):
    """Receives finished runs, for example to send them to an LLM tracing service.

    `export` must not block or raise: a sink that fails logs it and drops the run.
    """

    @property
    def capture_content(self) -> bool: ...

    def export(self, trace: RunTrace) -> None: ...

    def flush(self) -> None: ...


class NullTraceSink:
    """The default sink: it keeps nothing."""

    capture_content = False

    def export(self, trace: RunTrace) -> None:
        del trace

    def flush(self) -> None:
        return None


def export_run(
    sink: TraceSink,
    recorder: RunRecorder,
    *,
    name: str,
    input: str | None = None,
    output: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Send a finished run to `sink`; content goes only when the recorder captured it."""
    trace = RunTrace(
        name=name,
        recorder=recorder,
        ended_at=_now(),
        input=input if recorder.capture_content else None,
        output=output if recorder.capture_content else None,
        metadata=dict(metadata or {}),
    )
    try:
        sink.export(trace)
    except Exception as exc:  # a sink must never fail a request
        logger.warning("trace export failed: %s", type(exc).__name__)
