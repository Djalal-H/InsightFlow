"""FastAPI application entry point."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from insightflow.agents.checkpoint import open_checkpointer
from insightflow.api.dependencies import create_rate_limiter, create_tool_set, create_trace_sink
from insightflow.api.errors import (
    application_error_handler,
    provider_error_handler,
    storage_error_handler,
)
from insightflow.api.middleware import BodySizeLimitMiddleware, RequestContextMiddleware
from insightflow.api.routes.approvals import router as approvals_router
from insightflow.api.routes.health import router as health_router
from insightflow.api.routes.query import router as query_router
from insightflow.api.routes.threads import router as threads_router
from insightflow.core.config import get_settings
from insightflow.core.exceptions import InsightFlowError, ProviderError, StorageError
from insightflow.core.logging import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Configure logging, build the tool set and trace sink, and open the checkpointer once."""
    settings = get_settings()
    # Before the tool set: an in-process MCPServer configures root logging when none exists.
    configure_logging(
        settings.log_level, log_format=settings.log_format, secrets=settings.secret_values()
    )
    if settings.is_production and settings.langfuse_capture_content:
        logger.warning("LANGFUSE_CAPTURE_CONTENT is on in production: traces contain content")
    application.state.tool_set = create_tool_set(settings)
    sink = create_trace_sink(settings)
    application.state.trace_sink = sink
    limiter = create_rate_limiter(settings)
    application.state.rate_limiter = limiter
    try:
        async with open_checkpointer(
            settings.checkpoint_backend, settings.checkpoint_sqlite_path
        ) as checkpointer:
            application.state.checkpointer = checkpointer
            yield
    finally:
        sink.flush()
        if limiter is not None:
            await limiter.close()


def create_app() -> FastAPI:
    """Build the HTTP application.

    API docs (`/docs`, `/redoc`, `/openapi.json`) follow `API_DOCS_ENABLED` (default: off in
    production). The body limit runs inside the request context, so a refusal is logged.
    """
    settings = get_settings()
    docs = settings.docs_enabled
    application = FastAPI(
        title="InsightFlow API",
        version="0.1.0",
        description="API-first agent orchestration and retrieval platform.",
        lifespan=lifespan,
        docs_url="/docs" if docs else None,
        redoc_url="/redoc" if docs else None,
        openapi_url="/openapi.json" if docs else None,
    )
    application.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_request_bytes)
    application.add_middleware(RequestContextMiddleware)
    application.add_exception_handler(ProviderError, provider_error_handler)
    application.add_exception_handler(StorageError, storage_error_handler)
    application.add_exception_handler(InsightFlowError, application_error_handler)
    application.include_router(health_router)
    application.include_router(query_router)
    application.include_router(threads_router)
    application.include_router(approvals_router)
    return application


app = create_app()
