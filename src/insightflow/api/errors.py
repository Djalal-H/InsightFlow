"""Sanitized HTTP mappings for expected application failures."""

from typing import Final

from fastapi import Request, status
from fastapi.responses import JSONResponse

from insightflow.core.exceptions import (
    ProviderConfigurationError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)

CONFIGURATION_ERROR: Final = {
    "code": "provider_configuration_error",
    "message": "The chat provider is not configured correctly.",
}
TIMEOUT_ERROR: Final = {
    "code": "provider_timeout",
    "message": "The chat provider timed out.",
}
RATE_LIMIT_ERROR: Final = {
    "code": "provider_rate_limited",
    "message": "The chat provider is temporarily rate limited.",
}
UPSTREAM_ERROR: Final = {
    "code": "provider_error",
    "message": "The chat provider request failed.",
}


async def provider_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a fixed public response without exposing provider exception details."""
    del request

    if isinstance(exc, ProviderConfigurationError):
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        detail = CONFIGURATION_ERROR
    elif isinstance(exc, ProviderTimeoutError):
        status_code = status.HTTP_504_GATEWAY_TIMEOUT
        detail = TIMEOUT_ERROR
    elif isinstance(exc, ProviderRateLimitError):
        status_code = status.HTTP_429_TOO_MANY_REQUESTS
        detail = RATE_LIMIT_ERROR
    else:
        status_code = status.HTTP_502_BAD_GATEWAY
        detail = UPSTREAM_ERROR

    return JSONResponse(status_code=status_code, content={"error": detail})
