"""Sanitized HTTP mappings for expected application failures."""

from typing import Final

from fastapi import Request, status
from fastapi.responses import JSONResponse

from insightflow.core.exceptions import (
    InsightFlowError,
    ProviderConfigurationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    StorageError,
    ThreadNotFoundError,
)

CONFIGURATION_ERROR: Final = {
    "code": "provider_configuration_error",
    "message": "The model provider is not configured correctly.",
}
TIMEOUT_ERROR: Final = {
    "code": "provider_timeout",
    "message": "The model provider timed out.",
}
RATE_LIMIT_ERROR: Final = {
    "code": "provider_rate_limited",
    "message": "The model provider is temporarily rate limited.",
}
UPSTREAM_ERROR: Final = {
    "code": "provider_error",
    "message": "The model provider request failed.",
}
STORAGE_ERROR: Final = {
    "code": "storage_unavailable",
    "message": "The document store is unavailable.",
}


THREAD_NOT_FOUND_ERROR: Final = {
    "code": "thread_not_found",
    "message": "The conversation thread does not exist.",
}


def describe_error(exc: InsightFlowError) -> tuple[int, dict[str, str]]:
    """Return the HTTP status and fixed public detail for an expected application failure."""
    if isinstance(exc, StorageError):
        return status.HTTP_503_SERVICE_UNAVAILABLE, STORAGE_ERROR
    if isinstance(exc, ThreadNotFoundError):
        return status.HTTP_404_NOT_FOUND, THREAD_NOT_FOUND_ERROR
    if isinstance(exc, ProviderConfigurationError):
        return status.HTTP_503_SERVICE_UNAVAILABLE, CONFIGURATION_ERROR
    if isinstance(exc, ProviderTimeoutError):
        return status.HTTP_504_GATEWAY_TIMEOUT, TIMEOUT_ERROR
    if isinstance(exc, ProviderRateLimitError):
        return status.HTTP_429_TOO_MANY_REQUESTS, RATE_LIMIT_ERROR
    if isinstance(exc, ProviderError):
        return status.HTTP_502_BAD_GATEWAY, UPSTREAM_ERROR
    raise TypeError(f"no public mapping for {type(exc).__name__}")


async def provider_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a fixed public response without exposing provider exception details."""
    del request
    if not isinstance(exc, ProviderError):
        raise exc
    status_code, detail = describe_error(exc)
    return JSONResponse(status_code=status_code, content={"error": detail})


async def storage_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a fixed public response without exposing vector-store details."""
    del request
    if not isinstance(exc, StorageError):
        raise exc
    status_code, detail = describe_error(exc)
    return JSONResponse(status_code=status_code, content={"error": detail})


async def application_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return the fixed public response for any other expected application failure."""
    del request
    if not isinstance(exc, InsightFlowError):
        raise exc
    status_code, detail = describe_error(exc)
    return JSONResponse(status_code=status_code, content={"error": detail})
