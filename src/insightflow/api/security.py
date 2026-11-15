"""FastAPI dependencies that authenticate the caller and check scopes."""

from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import Depends, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from insightflow.api.dependencies import RateLimiterDependency, SettingsDependency
from insightflow.core.config import Settings, secret_text
from insightflow.core.exceptions import AuthenticationError, AuthorizationError
from insightflow.core.observability import current_recorder
from insightflow.security.audit import audit
from insightflow.security.auth import (
    LOCAL_PRINCIPAL,
    QUERY,
    THREADS_DELETE,
    THREADS_READ,
    JwtVerifier,
    Principal,
    create_jwt_verifier,
)

bearer_scheme = HTTPBearer(auto_error=False, description="A JWT access token.")


def _verifier_key(settings: Settings) -> tuple[object, ...]:
    return (
        secret_text(settings.auth_jwt_secret),
        settings.auth_jwks_url,
        settings.auth_jwt_algorithms,
        settings.auth_jwt_issuer,
        settings.auth_jwt_audience,
        settings.auth_jwt_leeway_seconds,
    )


def get_jwt_verifier(request: Request, settings: SettingsDependency) -> JwtVerifier | None:
    """One verifier per application and configuration, so the JWKS cache is shared."""
    if not settings.auth_enabled:
        return None
    key = _verifier_key(settings)
    cached: tuple[tuple[object, ...], JwtVerifier] | None = getattr(
        request.app.state, "jwt_verifier", None
    )
    if cached is not None and cached[0] == key:
        return cached[1]
    verifier = create_jwt_verifier(settings)
    assert verifier is not None
    request.app.state.jwt_verifier = (key, verifier)
    return verifier


async def get_principal(
    settings: SettingsDependency,
    verifier: Annotated[JwtVerifier | None, Depends(get_jwt_verifier)],
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> Principal:
    """The caller of the request; the local principal when authentication is off."""
    if not settings.auth_enabled or verifier is None:
        return LOCAL_PRINCIPAL
    if credentials is None or not credentials.credentials:
        audit("auth_failed", reason="missing_token")
        raise AuthenticationError("missing_token")
    try:
        principal = await verifier.verify(credentials.credentials)
    except AuthenticationError as exc:
        audit("auth_failed", reason=exc.reason)
        raise
    recorder = current_recorder()
    if recorder is not None:
        recorder.user = principal.user_key
    return principal


PrincipalDependency = Annotated[Principal, Depends(get_principal)]


def require_scopes(*scopes: str) -> Callable[[Principal], Awaitable[Principal]]:
    """A dependency that gives the principal when it has every scope, else 403."""

    async def dependency(principal: PrincipalDependency) -> Principal:
        for scope in scopes:
            if not principal.has(scope):
                audit("access_denied", missing_scope=scope)
                raise AuthorizationError(scope)
        return principal

    return dependency


QueryPrincipal = Annotated[Principal, Depends(require_scopes(QUERY))]
ThreadReader = Annotated[Principal, Depends(require_scopes(THREADS_READ))]
ThreadDeleter = Annotated[Principal, Depends(require_scopes(THREADS_DELETE))]


async def admit_query(
    principal: QueryPrincipal,
    limiter: RateLimiterDependency,
    response: Response,
) -> Principal:
    """A `query` principal whose request the rate limiter admits (429/503 otherwise).

    The `RateLimit-*` headers (IETF draft) report the burst that is left.
    """
    if limiter is None:
        return principal
    status = await limiter.acquire(principal.user_key)
    if status is not None:
        response.headers["RateLimit-Limit"] = str(status.limit)
        response.headers["RateLimit-Remaining"] = str(status.remaining)
        response.headers["RateLimit-Reset"] = str(status.reset_seconds)
    return principal


AdmittedPrincipal = Annotated[Principal, Depends(admit_query)]
