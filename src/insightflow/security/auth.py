"""Bearer-token authentication with JWTs and the scopes that authorize each operation.

Two key sources, chosen by settings: a shared HMAC secret (HS256/384/512, for local work and
tests) or an identity provider's JWKS endpoint (RS/ES/PS algorithms, for deployments). Only
the configured algorithms are accepted, so `alg: none` and HMAC-with-a-public-key confusion
are refused before any signature check.
"""

import asyncio
import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Final, Protocol

import jwt

from insightflow.core.config import Settings, secret_text
from insightflow.core.exceptions import AuthenticationError

QUERY: Final = "query"
THREADS_READ: Final = "threads:read"
THREADS_DELETE: Final = "threads:delete"
TOOLS_CONSEQUENTIAL: Final = "tools:consequential"
ADMIN: Final = "admin"
ALL_SCOPES: Final = frozenset({QUERY, THREADS_READ, THREADS_DELETE, TOOLS_CONSEQUENTIAL, ADMIN})
"""`admin` grants every other scope and access to threads of any owner."""

SUBJECT_MAX_LENGTH: Final = 256
JWKS_CACHE_SECONDS: Final = 300
JWKS_TIMEOUT_SECONDS: Final = 5


@dataclass(frozen=True)
class Principal:
    """The caller of one request: the token subject and its scopes."""

    subject: str
    scopes: frozenset[str]
    authenticated: bool = True
    """False for the local principal that is used when authentication is off."""

    def has(self, scope: str) -> bool:
        return scope in self.scopes or ADMIN in self.scopes

    @property
    def is_admin(self) -> bool:
        return ADMIN in self.scopes

    @property
    def owner_key(self) -> str:
        """The key stored as a thread's owner: the full SHA-256 of the subject."""
        return hashlib.sha256(self.subject.encode("utf-8")).hexdigest()

    def may_access(self, owner: str | None) -> bool:
        """Own threads only; administrators also reach other and ownerless threads."""
        return self.is_admin or (owner is not None and owner == self.owner_key)

    @property
    def user_key(self) -> str:
        """A stable pseudonym for logs and rate-limit keys; the subject is never logged."""
        return hashlib.sha256(self.subject.encode("utf-8")).hexdigest()[:16]


LOCAL_PRINCIPAL: Final = Principal(subject="local", scopes=ALL_SCOPES, authenticated=False)
"""The caller when `AUTH_ENABLED=false`: one local user with every scope."""


class SigningKeys(Protocol):
    """Finds the key that verifies a token's signature."""

    async def key_for(self, token: str) -> Any: ...


class SharedSecret:
    """One HMAC secret for every token."""

    def __init__(self, secret: str) -> None:
        self._secret = secret

    async def key_for(self, token: str) -> Any:
        del token
        return self._secret


class JwksKeys:
    """Keys from a JWKS endpoint, selected by the token's `kid` and cached by PyJWT."""

    def __init__(self, url: str, *, client: Any | None = None) -> None:
        self._client = client or jwt.PyJWKClient(
            url,
            cache_jwk_set=True,
            lifespan=JWKS_CACHE_SECONDS,
            timeout=JWKS_TIMEOUT_SECONDS,
        )

    async def key_for(self, token: str) -> Any:
        try:
            # The client uses blocking HTTP; keep it off the event loop.
            signing_key = await asyncio.to_thread(self._client.get_signing_key_from_jwt, token)
        except (jwt.PyJWKClientError, jwt.DecodeError) as exc:
            raise AuthenticationError("unknown_signing_key") from exc
        return signing_key.key


class JwtVerifier:
    """Verify a bearer token and return its principal; every failure is `AuthenticationError`."""

    def __init__(
        self,
        keys: SigningKeys,
        *,
        algorithms: Iterable[str],
        issuer: str = "",
        audience: str = "",
        leeway_seconds: int = 0,
    ) -> None:
        self._keys = keys
        self._algorithms = list(algorithms)
        if not self._algorithms or any(name.lower() == "none" for name in self._algorithms):
            raise ValueError("JWT verification needs an explicit algorithm allowlist")
        self._issuer = issuer.strip()
        self._audience = audience.strip()
        self._leeway = leeway_seconds

    async def verify(self, token: str) -> Principal:
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError as exc:
            raise AuthenticationError("malformed_token") from exc
        if header.get("alg") not in self._algorithms:
            raise AuthenticationError("algorithm_not_allowed")
        key = await self._keys.key_for(token)
        required = ["exp", "sub"]
        if self._issuer:
            required.append("iss")
        if self._audience:
            required.append("aud")
        try:
            claims = jwt.decode(
                token,
                key,
                algorithms=self._algorithms,
                issuer=self._issuer or None,
                audience=self._audience or None,
                leeway=self._leeway,
                options={"require": required, "verify_aud": bool(self._audience)},
            )
        except jwt.ExpiredSignatureError as exc:
            raise AuthenticationError("token_expired") from exc
        except jwt.ImmatureSignatureError as exc:
            raise AuthenticationError("token_not_yet_valid") from exc
        except jwt.InvalidIssuerError as exc:
            raise AuthenticationError("wrong_issuer") from exc
        except jwt.InvalidAudienceError as exc:
            raise AuthenticationError("wrong_audience") from exc
        except jwt.MissingRequiredClaimError as exc:
            raise AuthenticationError("missing_claim") from exc
        except jwt.PyJWTError as exc:
            raise AuthenticationError("invalid_token") from exc
        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject.strip() or len(subject) > SUBJECT_MAX_LENGTH:
            raise AuthenticationError("invalid_subject")
        return Principal(subject=subject, scopes=token_scopes(claims))


def token_scopes(claims: Mapping[str, Any]) -> frozenset[str]:
    """Scopes from `scope` (space-separated, RFC 8693) or the `scp`/`permissions` lists.

    Scopes that InsightFlow does not define are ignored.
    """
    found: set[str] = set()
    scope = claims.get("scope")
    if isinstance(scope, str):
        found.update(scope.split())
    for name in ("scp", "permissions"):
        value = claims.get(name)
        if isinstance(value, str):
            found.update(value.split())
        elif isinstance(value, list):
            found.update(item for item in value if isinstance(item, str))
    return frozenset(found & ALL_SCOPES)


def create_jwt_verifier(settings: Settings) -> JwtVerifier | None:
    """The verifier for the configured key source, or `None` when authentication is off."""
    if not settings.auth_enabled:
        return None
    secret = secret_text(settings.auth_jwt_secret)
    keys: SigningKeys = (
        SharedSecret(secret) if secret is not None else JwksKeys(settings.auth_jwks_url.strip())
    )
    return JwtVerifier(
        keys,
        algorithms=settings.jwt_algorithms,
        issuer=settings.auth_jwt_issuer,
        audience=settings.auth_jwt_audience,
        leeway_seconds=settings.auth_jwt_leeway_seconds,
    )
