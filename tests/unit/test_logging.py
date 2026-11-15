"""Tests for structured logging, run context fields, and secret redaction."""

import io
import json
import logging
from collections.abc import Iterator

import pytest

from insightflow.core.config import Settings
from insightflow.core.logging import (
    JsonFormatter,
    Redactor,
    TextFormatter,
    configure_logging,
)
from insightflow.core.observability import RunRecorder, node_scope, recording

SECRET = "nvapi-0123456789abcdef"


def emit(formatter: logging.Formatter, message: str, *args: object, **extra: object) -> str:
    record = logging.LogRecord("insightflow.test", logging.INFO, __file__, 1, message, args, None)
    for key, value in extra.items():
        setattr(record, key, value)
    return formatter.format(record)


def test_json_lines_carry_the_run_context_and_fields() -> None:
    formatter = JsonFormatter(Redactor())
    with recording(RunRecorder("req-7", thread_id="t-1")), node_scope("act"):
        line = emit(formatter, "provider call", fields={"latency_ms": 3.5, "outcome": "ok"})

    entry = json.loads(line)
    assert entry["message"] == "provider call"
    assert entry["level"] == "INFO"
    assert (entry["request_id"], entry["thread_id"], entry["node"]) == ("req-7", "t-1", "act")
    assert (entry["latency_ms"], entry["outcome"]) == (3.5, "ok")


def test_json_lines_outside_a_run_have_no_context_and_keep_exceptions() -> None:
    formatter = JsonFormatter(Redactor())
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = logging.LogRecord("x", logging.ERROR, __file__, 1, "failed", (), sys.exc_info())
    entry = json.loads(formatter.format(record))
    assert "request_id" not in entry
    assert "ValueError: boom" in entry["exception"]


def test_text_lines_append_context_and_fields() -> None:
    formatter = TextFormatter(Redactor())
    with recording(RunRecorder("req-8")):
        line = emit(formatter, "query completed", fields={"errors": {"provider_timeout": 1}})
    assert line.endswith('request_id=req-8 errors={"provider_timeout":1}')
    assert emit(formatter, "plain").endswith("plain")


@pytest.mark.parametrize("formatter_type", [JsonFormatter, TextFormatter])
def test_configured_secrets_and_token_shapes_are_redacted(
    formatter_type: type[JsonFormatter] | type[TextFormatter],
) -> None:
    formatter = formatter_type(Redactor(["my-configured-key-123", None, "short"]))
    line = emit(
        formatter,
        "auth %s with %s and Authorization: Bearer abc.def-ghi and tvly-dev-abcdefgh123",
        "my-configured-key-123",
        SECRET,
        fields={"header": "Bearer zzz"},
    )
    assert "my-configured-key-123" not in line
    assert SECRET not in line
    assert "abc.def-ghi" not in line
    assert "tvly-dev-abcdefgh123" not in line
    assert "zzz" not in line
    assert "[REDACTED]" in line


def test_short_values_are_not_treated_as_secrets() -> None:
    assert Redactor(["abc"])("abc stays") == "abc stays"


@pytest.fixture
def restore_root_logger() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


@pytest.mark.usefixtures("restore_root_logger")
def test_configure_logging_replaces_only_its_own_handler() -> None:
    root = logging.getLogger()
    foreign = logging.StreamHandler(io.StringIO())
    root.addHandler(foreign)

    configure_logging("debug", log_format="json", secrets=[SECRET])
    configure_logging("info", log_format="json", secrets=[SECRET])

    ours = [h for h in root.handlers if getattr(h, "_insightflow_handler", False)]
    assert len(ours) == 1
    assert isinstance(ours[0].formatter, JsonFormatter)
    assert foreign in root.handlers
    assert root.level == logging.INFO

    configure_logging()
    ours = [h for h in root.handlers if getattr(h, "_insightflow_handler", False)]
    assert isinstance(ours[0].formatter, TextFormatter)


def test_settings_list_every_configured_secret() -> None:
    settings = Settings(
        _env_file=None,
        litellm_api_key="llm-secret",
        litellm_rerank_api_key="rerank-secret",
        qdrant_api_key="qdrant-secret",
        tavily_api_key="tvly-secret",
        langfuse_enabled=True,
        langfuse_public_key="pk-lf-public",
        langfuse_secret_key="sk-lf-secret",
    )
    assert set(settings.secret_values()) == {
        "llm-secret",
        "rerank-secret",
        "qdrant-secret",
        "tvly-secret",
        "sk-lf-secret",
    }


def test_langfuse_needs_both_keys() -> None:
    with pytest.raises(ValueError, match="LANGFUSE_ENABLED"):
        Settings(
            _env_file=None,
            langfuse_enabled=True,
            langfuse_public_key="pk",
            langfuse_secret_key=None,
        )


def test_langfuse_host_accepts_the_sdk_variable_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")
    assert Settings(_env_file=None).langfuse_host == "https://us.cloud.langfuse.com"
    monkeypatch.setenv("LANGFUSE_HOST", "https://self.hosted.example")
    assert Settings(_env_file=None).langfuse_host == "https://self.hosted.example"
