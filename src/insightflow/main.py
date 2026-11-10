"""FastAPI application entry point."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from insightflow.agents.checkpoint import open_checkpointer
from insightflow.api.dependencies import create_tool_set
from insightflow.api.errors import (
    application_error_handler,
    provider_error_handler,
    storage_error_handler,
)
from insightflow.api.routes.health import router as health_router
from insightflow.api.routes.query import router as query_router
from insightflow.api.routes.threads import router as threads_router
from insightflow.core.config import get_settings
from insightflow.core.exceptions import InsightFlowError, ProviderError, StorageError
from insightflow.core.logging import configure_logging


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Configure logging, build the tool set, and open the workflow checkpointer once."""
    settings = get_settings()
    # Before the tool set: an in-process MCPServer configures root logging when none exists.
    configure_logging(settings.log_level)
    application.state.tool_set = create_tool_set(settings)
    async with open_checkpointer(
        settings.checkpoint_backend, settings.checkpoint_sqlite_path
    ) as checkpointer:
        application.state.checkpointer = checkpointer
        yield


def create_app() -> FastAPI:
    """Build the HTTP application."""
    application = FastAPI(
        title="InsightFlow API",
        version="0.1.0",
        description="API-first agent orchestration and retrieval platform.",
        lifespan=lifespan,
    )
    application.add_exception_handler(ProviderError, provider_error_handler)
    application.add_exception_handler(StorageError, storage_error_handler)
    application.add_exception_handler(InsightFlowError, application_error_handler)
    application.include_router(health_router)
    application.include_router(query_router)
    application.include_router(threads_router)
    return application


app = create_app()
