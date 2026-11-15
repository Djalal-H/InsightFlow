"""Provider and dependency failures, injected at each workflow step.

Each case checks the public contract: a fixed status and error code, no provider detail in the
body, the failure in the run's records, and a thread that the failed turn did not change.
"""

import logging
from typing import Any

import pytest

from insightflow.api.dependencies import get_checkpointer, get_rate_limiter
from insightflow.core.exceptions import (
    ProviderConfigurationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    StorageError,
)
from insightflow.security.rate_limit import RedisRateLimiter
from tests.fakes import FailingStreamProvider, ScriptedChatProvider, StaticRetriever
from tests.integration.test_query_api import client_for, make_app, memory_saver
from tests.integration.test_query_stream_api import stream_query
from tests.unit.test_rate_limit import BrokenRedis

SECRET_DETAIL = "upstream said: key nvapi-leaked-0123456789 is invalid"

# (failing role, error, expected status, expected code)
PROVIDER_FAULTS = [
    ("agent", ProviderTimeoutError(SECRET_DETAIL), 504, "provider_timeout"),
    ("agent", ProviderRateLimitError(SECRET_DETAIL), 429, "provider_rate_limited"),
    ("answer", ProviderError(SECRET_DETAIL), 502, "provider_error"),
    ("answer", ProviderConfigurationError(SECRET_DETAIL), 503, "provider_configuration_error"),
    ("judge", ProviderTimeoutError(SECRET_DETAIL), 504, "provider_timeout"),
]


def faulty_app(role: str, error: Exception, **settings: Any) -> Any:
    replies: dict[str, Any] = {role: error}
    if role == "judge":
        settings["agent_verifier_llm_enabled"] = True
    return make_app(ScriptedChatProvider(**replies), **settings)


@pytest.mark.parametrize(
    ("role", "error", "status", "code"),
    PROVIDER_FAULTS,
    ids=[f"{role}-{code}" for role, _, _, code in PROVIDER_FAULTS],
)
async def test_a_provider_failure_at_each_step_is_sanitized(
    role: str, error: Exception, status: int, code: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)
    async with client_for(faulty_app(role, error)) as client:
        response = await client.post("/query", json={"query": "Q"})

    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert "nvapi" not in response.text and "upstream" not in response.text
    failures = [r for r in caplog.records if r.getMessage() == "request failed"]
    assert [getattr(r, "fields", {}).get("code") for r in failures] == [code]


async def test_a_storage_failure_is_sanitized() -> None:
    application = make_app(
        ScriptedChatProvider(),
        StaticRetriever(error=StorageError(SECRET_DETAIL)),  # type: ignore[arg-type]
    )
    async with client_for(application) as client:
        response = await client.post("/query", json={"query": "Q"})
    assert (response.status_code, response.json()["error"]["code"]) == (
        503,
        "storage_unavailable",
    )
    assert "upstream" not in response.text


@pytest.mark.parametrize(("role", "error", "status", "code"), PROVIDER_FAULTS[:3])
async def test_a_failed_turn_leaves_the_thread_unchanged(
    role: str, error: Exception, status: int, code: str
) -> None:
    saver = memory_saver()
    good = make_app(ScriptedChatProvider())
    good.dependency_overrides[get_checkpointer] = lambda: saver
    async with client_for(good) as client:
        thread_id = (await client.post("/query", json={"query": "First"})).json()["thread_id"]

    bad = faulty_app(role, error)
    bad.dependency_overrides[get_checkpointer] = lambda: saver
    async with client_for(bad) as client:
        failed = await client.post("/query", json={"query": "Second", "thread_id": thread_id})
        history = await client.get(f"/threads/{thread_id}")

    assert failed.status_code == status
    assert [turn["query"] for turn in history.json()["turns"]] == ["First"]

    # The thread still works after the failure.
    async with client_for(good) as client:
        again = await client.post("/query", json={"query": "Third", "thread_id": thread_id})
        turns = (await client.get(f"/threads/{thread_id}")).json()["turns"]
    assert again.status_code == 200
    assert [turn["query"] for turn in turns] == ["First", "Third"]


async def test_a_stream_that_fails_after_some_tokens_ends_with_an_error_event() -> None:
    provider = FailingStreamProvider(ProviderError(SECRET_DETAIL), tokens_before_failure=2)
    status, _, events = await stream_query(make_app(provider), {"query": "Q"})

    names = [name for name, _ in events]
    assert status == 200
    assert names.count("token") == 2
    assert names[-1] == "error" and "final" not in names
    assert events[-1][1] == {
        "error": {"code": "provider_error", "message": "The model provider request failed."}
    }
    assert all("nvapi" not in str(data) for _, data in events)


async def test_an_unreachable_rate_limiter_admits_requests_when_failing_open() -> None:
    application = make_app(ScriptedChatProvider())
    limiter = RedisRateLimiter(BrokenRedis(), requests_per_minute=60, burst=5, fail_open=True)
    application.dependency_overrides[get_rate_limiter] = lambda: limiter
    async with client_for(application) as client:
        response = await client.post("/query", json={"query": "Q"})
    assert response.status_code == 200
    assert "ratelimit-limit" not in response.headers
