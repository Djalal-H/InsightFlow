"""Authentication, scopes, thread ownership, rate limits, and HTTP hardening through the API."""

from collections.abc import Sequence
from typing import Any

import pytest
from fakeredis import FakeAsyncRedis

from insightflow.agents.checkpoint import thread_config
from insightflow.api.dependencies import get_rate_limiter
from insightflow.core.config import get_settings
from insightflow.core.observability import report_usage
from insightflow.main import create_app
from insightflow.providers.llm import ChatMessage
from insightflow.security.rate_limit import RedisRateLimiter
from tests.fakes import AUTH_SETTINGS, ScriptedChatProvider, bearer, make_token
from tests.integration.test_query_api import client_for, make_app, memory_saver
from tests.unit.test_rate_limit import BrokenRedis

ALICE = bearer(make_token("alice"))
BOB = bearer(make_token("bob"))
ADMIN = bearer(make_token("ops", "admin"))


def secured(**options: Any) -> Any:
    return make_app(ScriptedChatProvider(), **{**AUTH_SETTINGS, **options})


async def test_health_needs_no_token() -> None:
    async with client_for(secured()) as client:
        assert (await client.get("/health/live")).status_code == 200


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Basic YWxpY2U6cHc="},
        bearer("not-a-jwt"),
        bearer(make_token(expires_in=-600)),
        bearer(make_token(aud="other")),
    ],
)
async def test_queries_without_a_valid_token_get_401(headers: dict[str, str]) -> None:
    async with client_for(secured()) as client:
        response = await client.post("/query", json={"query": "Q"}, headers=headers)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json() == {
        "error": {"code": "unauthorized", "message": "A valid bearer token is required."}
    }


async def test_a_token_without_the_scope_gets_403() -> None:
    async with client_for(secured()) as client:
        response = await client.post(
            "/query", json={"query": "Q"}, headers=bearer(make_token(scope="threads:read"))
        )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "forbidden"
    assert "insufficient_scope" in response.headers["www-authenticate"]


async def test_a_valid_token_runs_the_query() -> None:
    async with client_for(secured()) as client:
        response = await client.post("/query", json={"query": "Q"}, headers=ALICE)
    assert response.status_code == 200
    assert response.json()["status"] == "completed"


async def test_threads_belong_to_their_creator() -> None:
    application = secured(checkpointer=None)
    saver = memory_saver()
    from insightflow.api.dependencies import get_checkpointer

    application.dependency_overrides[get_checkpointer] = lambda: saver
    async with client_for(application) as client:
        created = await client.post("/query", json={"query": "Q"}, headers=ALICE)
        thread_id = created.json()["thread_id"]

        assert (await client.get(f"/threads/{thread_id}", headers=ALICE)).status_code == 200
        assert (await client.get(f"/threads/{thread_id}", headers=ADMIN)).status_code == 200
        # Another user cannot read, continue, or delete it, and cannot tell that it exists.
        foreign = await client.get(f"/threads/{thread_id}", headers=BOB)
        assert (foreign.status_code, foreign.json()["error"]["code"]) == (404, "thread_not_found")
        continued = await client.post(
            "/query", json={"query": "Q2", "thread_id": thread_id}, headers=BOB
        )
        assert continued.status_code == 404
        assert (await client.delete(f"/threads/{thread_id}", headers=BOB)).status_code == 404
        assert (await client.get(f"/threads/{thread_id}", headers=ALICE)).json()["turns"] == [
            {"query": "Q", "answer": "Grounded answer [1]."}
        ]

        # The owner continues it, and an administrator's turn does not take it over.
        await client.post("/query", json={"query": "Q2", "thread_id": thread_id}, headers=ALICE)
        await client.post("/query", json={"query": "Q3", "thread_id": thread_id}, headers=ADMIN)
        assert (await client.get(f"/threads/{thread_id}", headers=ALICE)).status_code == 200

        # Deleting needs its own scope.
        reader = bearer(make_token("alice", "query threads:read"))
        assert (await client.delete(f"/threads/{thread_id}", headers=reader)).status_code == 403
        assert (await client.delete(f"/threads/{thread_id}", headers=ALICE)).status_code == 204


async def test_threads_without_an_owner_are_open_only_to_administrators() -> None:
    application = secured()
    saver = memory_saver()
    from insightflow.api.dependencies import get_checkpointer

    application.dependency_overrides[get_checkpointer] = lambda: saver
    async with client_for(application) as client:
        created = await client.post("/query", json={"query": "Q"}, headers=ALICE)
        thread_id = created.json()["thread_id"]
    # Simulate a thread from before Stage 8: remove the owner from its checkpoint.
    stored = await saver.aget_tuple(thread_config(thread_id))
    assert stored is not None
    stored.checkpoint["channel_values"].pop("owner")
    await saver.aput(
        stored.config, stored.checkpoint, stored.metadata, stored.checkpoint["channel_versions"]
    )
    async with client_for(application) as client:
        assert (await client.get(f"/threads/{thread_id}", headers=ALICE)).status_code == 404
        assert (await client.get(f"/threads/{thread_id}", headers=ADMIN)).status_code == 200


class UsageReportingProvider(ScriptedChatProvider):
    """Reports token use like the LiteLLM adapter does."""

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        report_usage(10, 5)
        return await super().complete(messages)


def with_limiter(application: Any, limiter: Any) -> Any:
    application.dependency_overrides[get_rate_limiter] = lambda: limiter
    return application


async def test_the_rate_limit_answers_429_with_retry_after() -> None:
    limiter = RedisRateLimiter(
        FakeAsyncRedis(decode_responses=True), requests_per_minute=1, burst=2
    )
    async with client_for(with_limiter(secured(), limiter)) as client:
        first = await client.post("/query", json={"query": "Q"}, headers=ALICE)
        second = await client.post("/query", json={"query": "Q"}, headers=ALICE)
        third = await client.post("/query", json={"query": "Q"}, headers=ALICE)
        other_user = await client.post("/query", json={"query": "Q"}, headers=BOB)

    assert (first.status_code, second.status_code, other_user.status_code) == (200, 200, 200)
    assert (first.headers["ratelimit-limit"], first.headers["ratelimit-remaining"]) == ("2", "1")
    assert third.status_code == 429
    assert third.json()["error"]["code"] == "rate_limited"
    assert int(third.headers["retry-after"]) >= 1


async def test_the_token_quota_stops_a_user_after_the_reported_usage() -> None:
    limiter = RedisRateLimiter(
        FakeAsyncRedis(decode_responses=True),
        requests_per_minute=600,
        burst=10,
        daily_tokens=1,
    )
    application = with_limiter(make_app(UsageReportingProvider(), **AUTH_SETTINGS), limiter)
    async with client_for(application) as client:
        first = await client.post("/query", json={"query": "Q"}, headers=ALICE)
        second = await client.post("/query", json={"query": "Q"}, headers=ALICE)
    assert first.status_code == 200
    assert (second.status_code, second.json()["error"]["code"]) == (429, "quota_exceeded")


async def test_an_unreachable_rate_limiter_fails_closed() -> None:
    limiter = RedisRateLimiter(BrokenRedis(), requests_per_minute=60, burst=5)
    async with client_for(with_limiter(secured(), limiter)) as client:
        response = await client.post("/query", json={"query": "Q"}, headers=ALICE)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "rate_limiter_unavailable"


async def test_oversized_bodies_and_questions_are_refused() -> None:
    async with client_for(make_app(ScriptedChatProvider())) as client:
        huge = await client.post(
            "/query",
            content=b'{"query": "' + b"x" * 70_000 + b'"}',
            headers={"content-type": "application/json"},
        )
        long_question = await client.post("/query", json={"query": "x" * 4001})
    assert (huge.status_code, huge.json()["error"]["code"]) == (413, "payload_too_large")
    assert long_question.status_code == 422


async def test_responses_carry_security_headers() -> None:
    async with client_for(make_app(ScriptedChatProvider())) as client:
        query = await client.post("/query", json={"query": "Q"})
        health = await client.get("/health/live")
    assert query.headers["x-content-type-options"] == "nosniff"
    assert query.headers["cache-control"] == "no-store"
    assert "cache-control" not in health.headers


async def test_api_docs_can_be_turned_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_DOCS_ENABLED", "false")
    get_settings.cache_clear()
    try:
        application = create_app()
    finally:
        get_settings.cache_clear()
    async with client_for(application) as client:
        assert (await client.get("/openapi.json")).status_code == 404
        assert (await client.get("/docs")).status_code == 404
