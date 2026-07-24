"""FastAPI application entry point."""

from fastapi import FastAPI

from insightflow.api.errors import provider_error_handler
from insightflow.api.routes.health import router as health_router
from insightflow.api.routes.query import router as query_router
from insightflow.core.exceptions import ProviderError


def create_app() -> FastAPI:
    """Build the HTTP application."""
    application = FastAPI(
        title="InsightFlow API",
        version="0.1.0",
        description="API-first agent orchestration and retrieval platform.",
    )
    application.add_exception_handler(ProviderError, provider_error_handler)
    application.include_router(health_router)
    application.include_router(query_router)
    return application


app = create_app()
