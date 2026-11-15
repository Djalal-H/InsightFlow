"""ASGI middleware that opens one run record per HTTP request."""

import logging
import re
import time
import uuid
from typing import Final

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from insightflow.core.observability import RunRecorder, recording

logger = logging.getLogger("insightflow.access")

REQUEST_ID_HEADER: Final = "x-request-id"
REQUEST_ID_PATTERN: Final = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


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
