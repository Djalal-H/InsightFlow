"""FastAPI application entry point."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from insightflow.agents.checkpoint import open_checkpointer
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


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Open the workflow checkpointer once and close it at shutdown."""
    settings = get_settings()
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
