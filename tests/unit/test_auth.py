"""JWT verification, scopes, and principals."""

import base64
import json
import time
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from insightflow.core.config import Settings
from insightflow.core.exceptions import AuthenticationError
from insightflow.security.auth import (
    ADMIN,
    ALL_SCOPES,
    LOCAL_PRINCIPAL,
    QUERY,
    THREADS_READ,
    JwksKeys,
    JwtVerifier,
    Principal,
    SharedSecret,
    create_jwt_verifier,
    token_scopes,
)
from tests.fakes import AUTH_SETTINGS, TEST_AUDIENCE, TEST_ISSUER, TEST_JWT_SECRET, make_token


def verifier(**options: Any) -> JwtVerifier:
    values: dict[str, Any] = {
        "algorithms": ["HS256"],
        "issuer": TEST_ISSUER,
        "audience": TEST_AUDIENCE,
    }
    values.update(options)
    return JwtVerifier(SharedSecret(TEST_JWT_SECRET), **values)


async def reason(token: str, **options: Any) -> str:
    with pytest.raises(AuthenticationError) as error:
        await verifier(**options).verify(token)
    return error.value.reason


async def test_a_valid_token_gives_its_subject_and_known_scopes() -> None:
    principal = await verifier().verify(make_token("alice", "query threads:read unknown:scope"))
    assert principal.subject == "alice"
    assert principal.scopes == {QUERY, THREADS_READ}
    assert principal.authenticated


@pytest.mark.parametrize(
    ("token_options", "expected"),
    [
        ({"expires_in": -120}, "token_expired"),
        ({"secret": "another-secret-that-is-long-enough-012345"}, "invalid_token"),
        ({"iss": "https://other.test/"}, "wrong_issuer"),
        ({"aud": "other-service"}, "wrong_audience"),
        ({"nbf": int(time.time()) + 600}, "token_not_yet_valid"),
        ({"sub": None}, "missing_claim"),
        ({"sub": ""}, "invalid_subject"),
    ],
)
async def test_invalid_tokens_are_refused(token_options: dict[str, Any], expected: str) -> None:
    token = make_token(**{"subject": "alice", **token_options})
    if "sub" in token_options:
        payload = jwt.decode(token, options={"verify_signature": False})
        payload.pop("sub", None)
        if token_options["sub"] is not None:
            payload["sub"] = token_options["sub"]
        token = jwt.encode(payload, TEST_JWT_SECRET, "HS256")
    assert await reason(token) == expected


async def test_a_missing_expiry_is_refused() -> None:
    token = jwt.encode(
        {"sub": "alice", "iss": TEST_ISSUER, "aud": TEST_AUDIENCE}, TEST_JWT_SECRET, "HS256"
    )
    assert await reason(token) == "missing_claim"


async def test_the_leeway_accepts_a_token_that_just_expired() -> None:
    token = make_token(expires_in=-5)
    assert (await verifier(leeway_seconds=30).verify(token)).subject == "alice"


def _unsigned(header: dict[str, Any], payload: dict[str, Any]) -> str:
    def part(value: dict[str, Any]) -> str:
        return base64.urlsafe_b64encode(json.dumps(value).encode()).rstrip(b"=").decode()

    return f"{part(header)}.{part(payload)}."


async def test_alg_none_and_other_algorithms_are_refused() -> None:
    claims = {"sub": "x", "exp": int(time.time()) + 60, "iss": TEST_ISSUER, "aud": TEST_AUDIENCE}
    assert await reason(_unsigned({"alg": "none", "typ": "JWT"}, claims)) == "algorithm_not_allowed"
    hs512 = jwt.encode(claims, TEST_JWT_SECRET + "x" * 32, "HS512")
    assert await reason(hs512) == "algorithm_not_allowed"
    assert await reason("not a token") == "malformed_token"


def test_an_empty_or_none_allowlist_is_a_programming_error() -> None:
    with pytest.raises(ValueError):
        JwtVerifier(SharedSecret("x"), algorithms=[])
    with pytest.raises(ValueError):
        JwtVerifier(SharedSecret("x"), algorithms=["none"])


class FakeJwksClient:
    """Stands in for `PyJWKClient`: one RSA key, selected by `kid`."""

    def __init__(self, key: rsa.RSAPrivateKey, kid: str = "key-1") -> None:
        self.public_key = key.public_key()
        self.kid = kid
        self.calls = 0

    def get_signing_key_from_jwt(self, token: str) -> Any:
        self.calls += 1
        if jwt.get_unverified_header(token).get("kid") != self.kid:
            raise jwt.PyJWKClientError("Unable to find a signing key")

        class Key:
            key = self.public_key

        return Key()


async def test_rs256_tokens_are_verified_with_the_jwks_key() -> None:
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    client = FakeJwksClient(private)
    rs = JwtVerifier(
        JwksKeys("https://issuer.test/jwks", client=client),
        algorithms=["RS256"],
        issuer=TEST_ISSUER,
        audience=TEST_AUDIENCE,
    )
    claims = {
        "sub": "bob",
        "scp": ["query"],
        "exp": int(time.time()) + 60,
        "iss": TEST_ISSUER,
        "aud": TEST_AUDIENCE,
    }
    token = jwt.encode(claims, private, "RS256", headers={"kid": "key-1"})
    assert (await rs.verify(token)).scopes == {QUERY}

    unknown = jwt.encode(claims, private, "RS256", headers={"kid": "rotated"})
    with pytest.raises(AuthenticationError) as error:
        await rs.verify(unknown)
    assert error.value.reason == "unknown_signing_key"

    # HMAC with a public-key-shaped secret: refused by the allowlist before any key use.
    forged = jwt.encode(claims, TEST_JWT_SECRET, "HS256", headers={"kid": "key-1"})
    with pytest.raises(AuthenticationError):
        await rs.verify(forged)


def test_scope_claims_in_several_shapes() -> None:
    assert token_scopes({"scope": "query admin"}) == {QUERY, ADMIN}
    assert token_scopes({"scp": "threads:read"}) == {THREADS_READ}
    assert token_scopes({"permissions": ["query", 7, "nope"]}) == {QUERY}
    assert token_scopes({}) == frozenset()


def test_principals_and_thread_access() -> None:
    alice = Principal("alice", frozenset({QUERY}))
    admin = Principal("root", frozenset({ADMIN}))
    assert alice.has(QUERY) and not alice.has(THREADS_READ)
    assert admin.has(THREADS_READ)
    assert alice.may_access(alice.owner_key)
    assert not alice.may_access(admin.owner_key)
    assert not alice.may_access(None)
    assert admin.may_access(alice.owner_key) and admin.may_access(None)
    assert len(alice.user_key) == 16 and alice.user_key != alice.subject
    assert len(alice.owner_key) == 64
    assert LOCAL_PRINCIPAL.scopes == ALL_SCOPES and not LOCAL_PRINCIPAL.authenticated


def test_the_verifier_follows_the_settings() -> None:
    assert create_jwt_verifier(Settings(_env_file=None)) is None
    assert isinstance(create_jwt_verifier(Settings(_env_file=None, **AUTH_SETTINGS)), JwtVerifier)
