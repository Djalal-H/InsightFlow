"""GCRA rate limits and daily quotas against fakeredis (Lua included) and a real Redis."""

import asyncio
import os
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fakeredis import FakeAsyncRedis
from redis.exceptions import ConnectionError as RedisConnectionError

from insightflow.core.exceptions import (
    QuotaExceededError,
    RateLimiterUnavailableError,
    RateLimitExceededError,
)
from insightflow.security.rate_limit import KEY_PREFIX, RedisRateLimiter

# Noon tomorrow (UTC): the quota keys expire at the end of that day, which is always in the
# future for the real clock that fakeredis uses for EXPIREAT.
NOON = datetime.now(UTC).replace(hour=12, minute=0, second=0, microsecond=0) + timedelta(days=1)
TODAY = NOON.strftime("%Y%m%d")


def limiter(client: Any | None = None, **values: Any) -> RedisRateLimiter:
    options: dict[str, Any] = {"requests_per_minute": 60, "burst": 3, "clock": lambda: NOON}
    options.update(values)
    return RedisRateLimiter(client or FakeAsyncRedis(decode_responses=True), **options)


async def test_a_burst_is_admitted_and_the_next_request_waits() -> None:
    rate = limiter()
    statuses = [await rate.acquire("user-a") for _ in range(3)]
    assert [status.remaining for status in statuses if status] == [2, 1, 0]

    with pytest.raises(RateLimitExceededError) as error:
        await rate.acquire("user-a")
    assert error.value.retry_after_seconds == 1
    assert error.value.limit == 3


async def test_users_have_separate_limits() -> None:
    rate = limiter(burst=1)
    await rate.acquire("user-a")
    assert await rate.acquire("user-b") is not None
    with pytest.raises(RateLimitExceededError):
        await rate.acquire("user-a")


async def test_the_rate_recovers_over_time() -> None:
    rate = limiter(requests_per_minute=6000, burst=1)  # one request per 10 ms
    await rate.acquire("user-a")
    await asyncio.sleep(0.03)
    assert await rate.acquire("user-a") is not None


async def test_the_daily_request_quota_stops_requests_until_tomorrow() -> None:
    rate = limiter(burst=10, daily_requests=2)
    await rate.acquire("user-a")
    await rate.acquire("user-a")
    with pytest.raises(QuotaExceededError) as error:
        await rate.acquire("user-a")
    assert error.value.quota == "requests"
    assert error.value.retry_after_seconds == 12 * 3600


async def test_a_refused_request_does_not_use_the_request_quota() -> None:
    client = FakeAsyncRedis(decode_responses=True)
    rate = limiter(client, burst=1, daily_requests=5)
    await rate.acquire("user-a")
    with pytest.raises(RateLimitExceededError):
        await rate.acquire("user-a")
    assert await client.get(f"{KEY_PREFIX}:requests:{TODAY}:user-a") == "1"


async def test_the_token_quota_counts_charged_tokens() -> None:
    client = FakeAsyncRedis(decode_responses=True)
    rate = limiter(client, burst=10, daily_tokens=100)
    await rate.acquire("user-a")
    await rate.charge_tokens("user-a", 60)
    await rate.acquire("user-a")
    await rate.charge_tokens("user-a", 60)

    with pytest.raises(QuotaExceededError) as error:
        await rate.acquire("user-a")
    assert error.value.quota == "tokens"
    key = f"{KEY_PREFIX}:tokens:{TODAY}:user-a"
    assert await client.get(key) == "120"
    assert await client.ttl(key) > 0


async def test_no_token_quota_charges_nothing() -> None:
    client = FakeAsyncRedis(decode_responses=True)
    await limiter(client).charge_tokens("user-a", 500)
    assert await client.keys("*") == []


class BrokenRedis(FakeAsyncRedis):
    def register_script(self, script: str) -> Any:
        async def fail(**_: Any) -> Any:
            raise RedisConnectionError("down")

        return fail

    async def ping(self, **_: Any) -> bool:
        raise RedisConnectionError("down")


async def test_an_unreachable_redis_fails_closed_by_default() -> None:
    rate = limiter(BrokenRedis())
    with pytest.raises(RateLimiterUnavailableError):
        await rate.acquire("user-a")
    assert await rate.ping() is False


async def test_an_unreachable_redis_admits_requests_when_failing_open() -> None:
    rate = limiter(BrokenRedis(), fail_open=True)
    assert await rate.acquire("user-a") is None
    await rate.charge_tokens("user-a", 10)  # never raises


def test_limits_must_be_positive() -> None:
    with pytest.raises(ValueError):
        limiter(requests_per_minute=0)


@pytest.mark.redis
async def test_the_script_runs_on_a_real_redis() -> None:
    url = os.environ.get("REDIS_TEST_URL")
    if not url:
        pytest.skip("REDIS_TEST_URL is not set")
    from insightflow.storage.redis import create_redis_client

    client = create_redis_client(url)
    user = f"test-{os.getpid()}-{datetime.now(UTC).timestamp()}"
    try:
        rate = limiter(client, burst=2, daily_requests=10, clock=lambda: datetime.now(UTC))
        assert await rate.ping()
        await rate.acquire(user)
        await rate.acquire(user)
        with pytest.raises(RateLimitExceededError):
            await rate.acquire(user)
    finally:
        await client.delete(*await client.keys(f"{KEY_PREFIX}:*{user}"))
        await client.aclose()
