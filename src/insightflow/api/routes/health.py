"""Liveness and readiness endpoints."""

from fastapi import APIRouter, Response, status

from insightflow.api.schemas import HealthResponse
from insightflow.core.config import get_settings
from insightflow.storage.qdrant import QdrantStore

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live", response_model=HealthResponse)
async def liveness() -> HealthResponse:
    """Report whether the API process is serving requests."""
    return HealthResponse(status="ok", checks={"api": "available"})


@router.get("/ready", response_model=HealthResponse)
async def readiness(response: Response) -> HealthResponse:
    """Report whether required model configuration and Qdrant are available."""
    settings = get_settings()
    checks = {
        "chat_model": "configured" if settings.litellm_chat_model else "missing",
        "embedding_model": "configured" if settings.litellm_embedding_model else "missing",
    }

    store = QdrantStore(settings)
    checks["qdrant"] = "available" if await store.is_available() else "unavailable"
    await store.close()

    is_ready = all(value in {"configured", "available"} for value in checks.values())
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(status="ok" if is_ready else "not_ready", checks=checks)

