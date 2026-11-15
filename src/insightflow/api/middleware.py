"""ASGI middleware: one run record per HTTP request, security headers, and a body limit."""

import json
import logging
import re
import time
import uuid
from typing import Final

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from insightflow.api.errors import PAYLOAD_TOO_LARGE_ERROR
from insightflow.core.observability import RunRecorder, recording

logger = logging.getLogger("insightflow.access")

REQUEST_ID_HEADER: Final = "x-request-id"
REQUEST_ID_PATTERN: Final = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
NO_STORE_PREFIXES: Final = ("/query", "/threads")
"""Responses with conversation content: clients and proxies must not cache them."""


def request_id_from(scope: Scope) -> str:
    """The client's `X-Request-ID` when it is a safe token, else a new identifier."""
    for name, value in scope.get("headers", []):
        if name.decode("latin-1").lower() == REQUEST_ID_HEADER:
            candidate = bytes(value).decode("latin-1")
            if REQUEST_ID_PATTERN.fullmatch(candidate):
                return candidate
            break
    return uuid.uuid4().hex


class RequestContextMiddleware:
    """Give each request a request ID and a `RunRecorder`, return the ID, log one access line.

    The recorder is active for the whole request, so dependencies, the workflow, and streamed
    responses report to it. Request and response bodies are never logged.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        recorder = RunRecorder(request_id_from(scope))
        started = time.perf_counter()
        status_code = 500

        async def send_with_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = list(message.get("headers", []))
                headers.append((REQUEST_ID_HEADER.encode(), recorder.request_id.encode()))
                headers.append((b"x-content-type-options", b"nosniff"))
                if str(scope.get("path", "")).startswith(NO_STORE_PREFIXES):
                    headers.append((b"cache-control", b"no-store"))
                message = {**message, "headers": headers}
            await send(message)

        with recording(recorder):
            try:
                await self.app(scope, receive, send_with_id)
            finally:
                logger.info(
                    "request",
                    extra={
                        "fields": {
                            "method": scope.get("method"),
                            "path": scope.get("path"),
                            "status": status_code,
                            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                        }
                    },
                )


class _BodyTooLarge(Exception):
    pass


class BodySizeLimitMiddleware:
    """Refuse request bodies larger than `max_bytes` with 413, before the app reads them all.

    A `Content-Length` over the limit is refused at once; a body without one (chunked) is
    counted while it is read.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        for name, value in scope.get("headers", []):
            if name.lower() == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    declared = 0
                if declared > self.max_bytes:
                    await _send_too_large(send)
                    return
        received = 0
        started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLarge
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _BodyTooLarge:
            if not started:
                await _send_too_large(send)


async def _send_too_large(send: Send) -> None:
    body = json.dumps({"error": PAYLOAD_TOO_LARGE_ERROR}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
