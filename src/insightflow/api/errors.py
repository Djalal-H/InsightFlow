"""Sanitized HTTP mappings for expected application failures."""

import logging
from typing import Final

from fastapi import Request, status
from fastapi.responses import JSONResponse

from insightflow.core.exceptions import (
    ApprovalNotFoundError,
    AuthenticationError,
    AuthorizationError,
    InputRejectedError,
    InsightFlowError,
    ProviderConfigurationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    QuotaExceededError,
    RateLimiterUnavailableError,
    RateLimitExceededError,
    StorageError,
    ThreadNotFoundError,
)
from insightflow.core.observability import record_error

logger = logging.getLogger(__name__)

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
UNAUTHORIZED_ERROR: Final = {
    "code": "unauthorized",
    "message": "A valid bearer token is required.",
}
FORBIDDEN_ERROR: Final = {
    "code": "forbidden",
    "message": "The token does not allow this operation.",
}
RATE_LIMITED_ERROR: Final = {
    "code": "rate_limited",
    "message": "Too many requests. Wait before you send the next one.",
}
QUOTA_EXCEEDED_ERROR: Final = {
    "code": "quota_exceeded",
    "message": "The daily usage quota is used up.",
}
RATE_LIMITER_UNAVAILABLE_ERROR: Final = {
    "code": "rate_limiter_unavailable",
    "message": "Requests cannot be accepted at the moment.",
}
INPUT_REJECTED_ERROR: Final = {
    "code": "input_rejected",
    "message": "The request was refused by a content policy.",
}
APPROVAL_NOT_FOUND_ERROR: Final = {
    "code": "approval_not_found",
    "message": "The thread has no pending approval with this identifier.",
}
PAYLOAD_TOO_LARGE_ERROR: Final = {
    "code": "payload_too_large",
    "message": "The request body is too large.",
}


def describe_error(exc: InsightFlowError) -> tuple[int, dict[str, str]]:
    """Return the HTTP status and fixed public detail for an expected application failure."""
    if isinstance(exc, AuthenticationError):
        return status.HTTP_401_UNAUTHORIZED, UNAUTHORIZED_ERROR
    if isinstance(exc, AuthorizationError):
        return status.HTTP_403_FORBIDDEN, FORBIDDEN_ERROR
    if isinstance(exc, RateLimitExceededError):
        return status.HTTP_429_TOO_MANY_REQUESTS, RATE_LIMITED_ERROR
    if isinstance(exc, QuotaExceededError):
        return status.HTTP_429_TOO_MANY_REQUESTS, QUOTA_EXCEEDED_ERROR
    if isinstance(exc, RateLimiterUnavailableError):
        return status.HTTP_503_SERVICE_UNAVAILABLE, RATE_LIMITER_UNAVAILABLE_ERROR
    if isinstance(exc, InputRejectedError):
        return status.HTTP_400_BAD_REQUEST, INPUT_REJECTED_ERROR
    if isinstance(exc, ApprovalNotFoundError):
        return status.HTTP_404_NOT_FOUND, APPROVAL_NOT_FOUND_ERROR
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


def error_headers(exc: InsightFlowError) -> dict[str, str]:
    """The HTTP headers that a failure needs (RFC 6750 challenge, RFC 9110 Retry-After)."""
    if isinstance(exc, AuthenticationError):
        return {"WWW-Authenticate": "Bearer"}
    if isinstance(exc, AuthorizationError):
        return {"WWW-Authenticate": 'Bearer error="insufficient_scope"'}
    if isinstance(exc, RateLimitExceededError | QuotaExceededError):
        return {"Retry-After": str(exc.retry_after_seconds)}
    return {}


def report_error(exc: InsightFlowError) -> tuple[int, dict[str, str]]:
    """Map a failure like `describe_error`, and log and record its code (not its message)."""
    status_code, detail = describe_error(exc)
    record_error(detail["code"])
    logger.warning(
        "request failed",
        extra={"fields": {"code": detail["code"], "error_type": type(exc).__name__}},
    )
    return status_code, detail


async def provider_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a fixed public response without exposing provider exception details."""
    del request
    if not isinstance(exc, ProviderError):
        raise exc
    status_code, detail = report_error(exc)
    return JSONResponse(
        status_code=status_code, content={"error": detail}, headers=error_headers(exc)
    )


async def storage_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a fixed public response without exposing vector-store details."""
    del request
    if not isinstance(exc, StorageError):
        raise exc
    status_code, detail = report_error(exc)
    return JSONResponse(
        status_code=status_code, content={"error": detail}, headers=error_headers(exc)
    )


async def application_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return the fixed public response for any other expected application failure."""
    del request
    if not isinstance(exc, InsightFlowError):
        raise exc
    status_code, detail = report_error(exc)
    return JSONResponse(
        status_code=status_code, content={"error": detail}, headers=error_headers(exc)
    )
