"""Standard-library logging configuration: text or JSON lines, request context, redaction.

Every record gets the `request_id`, `thread_id`, `user` (a hash), and workflow `node` of the
active run (see
`core.observability`). Structured values are passed as `extra={"fields": {...}}`. The formatted
line is redacted: configured secrets and bearer tokens never reach the output.
"""

import json
import logging
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any, Literal

from insightflow.core.observability import current_node, current_recorder

LogFormat = Literal["text", "json"]
REDACTED = "[REDACTED]"
_HANDLER_MARK = "_insightflow_handler"
_TOKEN_PATTERNS = (
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"\b(?:tvly|nvapi|sk-lf|pk-lf|sk)-[A-Za-z0-9_-]{8,}"),
    # A JWT (header.payload.signature), for example an access token in an error message.
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]*"),
)
QUIET_LOGGERS = ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy", "httpx", "httpcore", "openai")
"""Third-party loggers that can write request or response bodies at DEBUG; kept at WARNING."""
_MIN_SECRET_LENGTH = 6


def context_fields() -> dict[str, Any]:
    """The run context of the current task, or nothing outside a run."""
    recorder = current_recorder()
    fields: dict[str, Any] = {}
    if recorder is not None:
        fields["request_id"] = recorder.request_id
        if recorder.thread_id is not None:
            fields["thread_id"] = recorder.thread_id
        if recorder.user is not None:
            fields["user"] = recorder.user
    node = current_node()
    if node is not None:
        fields["node"] = node
    return fields


class Redactor:
    """Mask configured secret values and well-known token shapes in a string."""

    def __init__(self, secrets: Iterable[str | None] = ()) -> None:
        values = {value for value in secrets if value and len(value) >= _MIN_SECRET_LENGTH}
        # Longest first, so a secret that contains another is masked whole.
        self._secrets = sorted(values, key=len, reverse=True)

    def __call__(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, REDACTED)
        for pattern in _TOKEN_PATTERNS:
            text = pattern.sub(_mask_token, text)
        return text


def _mask_token(match: re.Match[str]) -> str:
    value = match.group(0)
    if value.lower().startswith("bearer"):
        return f"Bearer {REDACTED}"
    return REDACTED


def _record_fields(record: logging.LogRecord) -> dict[str, Any]:
    fields = getattr(record, "fields", None)
    return dict(fields) if isinstance(fields, dict) else {}


class TextFormatter(logging.Formatter):
    """`<time> <level> <logger> <message> key=value ...`, redacted."""

    def __init__(self, redactor: Redactor) -> None:
        super().__init__("%(asctime)s %(levelname)s %(name)s %(message)s")
        self._redact = redactor

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        fields = {**context_fields(), **_record_fields(record)}
        if fields:
            pairs = " ".join(f"{key}={_text_value(value)}" for key, value in fields.items())
            line = f"{line} {pairs}"
        return self._redact(line)


def _text_value(value: Any) -> str:
    if isinstance(value, dict | list):
        return json.dumps(value, separators=(",", ":"), default=str)
    return str(value)


class JsonFormatter(logging.Formatter):
    """One JSON object per line, redacted."""

    def __init__(self, redactor: Redactor) -> None:
        super().__init__()
        self._redact = redactor

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            **context_fields(),
            **_record_fields(record),
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return self._redact(json.dumps(entry, default=str, ensure_ascii=False))


def configure_logging(
    level: str = "INFO",
    *,
    log_format: LogFormat = "text",
    secrets: Iterable[str | None] = (),
) -> None:
    """Install one root handler with the chosen format; calling it again replaces it.

    Handlers that other code installed are left in place. Call this before code that calls
    `logging.basicConfig` (an in-process MCPServer does), so that call finds a handler and
    changes nothing.
    """
    redactor = Redactor(secrets)
    formatter: logging.Formatter = (
        JsonFormatter(redactor) if log_format == "json" else TextFormatter(redactor)
    )
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_MARK, False):
            root.removeHandler(handler)
    handler = logging.StreamHandler()
    setattr(handler, _HANDLER_MARK, True)
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(level.upper())
    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
