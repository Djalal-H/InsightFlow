"""FastAPI application entry point."""

from fastapi import FastAPI

from insightflow.api.routes.health import router as health_router


def create_app() -> FastAPI:
    """Build the HTTP application."""
    application = FastAPI(
        title="InsightFlow API",
        version="0.1.0",
        description="API-first agent orchestration and retrieval platform.",
    )
    application.include_router(health_router)
    return application


app = create_app()

