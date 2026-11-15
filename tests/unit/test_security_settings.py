"""Stage 8 settings: secrets, authentication modes, rate limits, and the production profile."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from insightflow.core.config import Settings, get_settings, secret_text

HS_SECRET = "s" * 32
JWKS = "https://issuer.test/jwks"
PRODUCTION: dict[str, Any] = {
    "app_env": "production",
    "auth_enabled": True,
    "auth_jwks_url": "https://issuer.test/.well-known/jwks.json",
    "auth_jwt_issuer": "https://issuer.test/",
    "auth_jwt_audience": "insightflow",
    "rate_limit_enabled": True,
    "redis_url": "redis://:redis-password@redis:6379/0",
    "log_format": "json",
}


def settings(**values: Any) -> Settings:
    return Settings(_env_file=None, **values)


def test_api_keys_are_secret_strings_and_hidden_in_repr() -> None:
    configured = settings(
        litellm_api_key="llm-secret-value",
        qdrant_api_key="qdrant-secret-value",
        litellm_rerank_api_key="rerank-secret-value",
    )
    assert isinstance(configured.litellm_api_key, SecretStr)
    assert "llm-secret-value" not in repr(configured)
    assert "qdrant-secret-value" not in repr(configured)
    assert "rerank-secret-value" not in repr(configured)


def test_secret_text_treats_blank_values_as_unset() -> None:
    assert secret_text(None) is None
    assert secret_text(SecretStr("  ")) is None
    assert secret_text(SecretStr("value")) == "value"


def test_secret_values_include_the_auth_secret_and_the_redis_password() -> None:
    configured = settings(
        auth_jwt_secret=HS_SECRET,
        redis_url="redis://:redis-password@redis:6379/0",
        guard_safety_api_key="guard-secret",
    )
    values = set(configured.secret_values())
    assert {HS_SECRET, "redis-password", "guard-secret"} <= values
    assert "redis://:redis-password@redis:6379/0" in values


def test_auth_needs_exactly_one_key_source() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        settings(auth_enabled=True)
    with pytest.raises(ValueError, match="exactly one"):
        settings(
            auth_enabled=True,
            auth_jwt_secret=HS_SECRET,
            auth_jwks_url="https://issuer.test/jwks",
            auth_jwt_algorithms="HS256",
        )


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"auth_jwt_secret": HS_SECRET, "auth_jwt_algorithms": "RS256"}, "only HS256"),
        ({"auth_jwks_url": JWKS, "auth_jwt_algorithms": "HS256"}, "asymmetric"),
        ({"auth_jwt_secret": "short", "auth_jwt_algorithms": "HS256"}, "32 characters"),
        ({"auth_jwks_url": JWKS, "auth_jwt_algorithms": "none"}, "none"),
        ({"auth_jwks_url": JWKS, "auth_jwt_algorithms": " , "}, "at least one"),
    ],
)
def test_incoherent_auth_settings_are_refused(values: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        settings(auth_enabled=True, **values)


def test_valid_auth_modes_are_accepted() -> None:
    symmetric = settings(auth_enabled=True, auth_jwt_secret=HS_SECRET, auth_jwt_algorithms="HS256")
    assert symmetric.jwt_algorithms == ["HS256"]
    asymmetric = settings(
        auth_enabled=True,
        auth_jwks_url="https://issuer.test/jwks",
        auth_jwt_algorithms="RS256, ES256",
    )
    assert asymmetric.jwt_algorithms == ["RS256", "ES256"]


def test_rate_limits_need_redis() -> None:
    with pytest.raises(ValueError, match="REDIS_URL"):
        settings(rate_limit_enabled=True)


def test_production_requires_the_protections() -> None:
    with pytest.raises(ValueError) as error:
        settings(app_env="production")
    message = str(error.value)
    for name in ("AUTH_ENABLED", "AUTH_JWT_ISSUER", "RATE_LIMIT_ENABLED", "LOG_FORMAT=json"):
        assert name in message


def test_a_complete_production_profile_turns_api_docs_off() -> None:
    production = settings(**PRODUCTION)
    assert production.is_production
    assert production.docs_enabled is False
    assert settings(**PRODUCTION, api_docs_enabled=True).docs_enabled is True
    assert settings().docs_enabled is True


def test_secrets_dir_supplies_secret_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    (secrets / "litellm_api_key").write_text("from-a-secret-file")
    # Environment variables and `.env` win over secret files: use an empty directory.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SECRETS_DIR", str(secrets))
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    get_settings.cache_clear()
    try:
        loaded = get_settings()
    finally:
        get_settings.cache_clear()
    assert secret_text(loaded.litellm_api_key) == "from-a-secret-file"
