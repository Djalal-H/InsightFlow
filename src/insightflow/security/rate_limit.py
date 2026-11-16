"""Per-user rate limits and daily quotas in Redis.

The rate is a GCRA (generic cell rate algorithm) limit: `requests_per_minute` on average, with
up to `burst` requests at once. Quotas count requests and model tokens per user and UTC day.
One Lua script checks the token quota, the request quota, and the rate, and records the request
only when all three allow it, so concurrent requests on several app instances cannot pass a
limit together. Redis' own clock is used, so instances with different clocks agree.
"""

import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final, Protocol

from redis.exceptions import RedisError

from insightflow.core.exceptions import (
    QuotaExceededError,
    RateLimiterUnavailableError,
    RateLimitExceededError,
)
from insightflow.security.audit import audit

logger = logging.getLogger(__name__)

KEY_PREFIX: Final = "insightflow:rl"
QUOTA_KEY_GRACE_SECONDS: Final = 3600
"""Quota keys live one hour past the end of their day, then Redis removes them."""

# KEYS: 1 rate (theoretical arrival time, ms), 2 request count, 3 token count.
# ARGV: 1 emission interval (ms), 2 burst tolerance (ms), 3 request quota (0 = off),
#       4 token quota (0 = off), 5 quota key expiry (unix seconds).
# Returns {status, a, b}: status 1 allowed (a = remaining burst, b = ms until the full burst
# is back); 0 rate limited (a = ms to wait); 2 request quota used up; 3 token quota used up.
ACQUIRE_SCRIPT: Final = """
local time = redis.call('TIME')
local now = tonumber(time[1]) * 1000 + math.floor(tonumber(time[2]) / 1000)
local interval = tonumber(ARGV[1])
local tolerance = tonumber(ARGV[2])
local request_quota = tonumber(ARGV[3])
local token_quota = tonumber(ARGV[4])
if token_quota > 0 and tonumber(redis.call('GET', KEYS[3]) or '0') >= token_quota then
  return {3, 0, 0}
end
if request_quota > 0 and tonumber(redis.call('GET', KEYS[2]) or '0') >= request_quota then
  return {2, 0, 0}
end
local tat = tonumber(redis.call('GET', KEYS[1]) or now)
if tat < now then tat = now end
local new_tat = tat + interval
local allow_at = new_tat - tolerance
if now < allow_at then
  return {0, allow_at - now, 0}
end
redis.call('SET', KEYS[1], new_tat, 'PX', math.max(1, new_tat - now))
if request_quota > 0 then
  redis.call('INCR', KEYS[2])
  redis.call('EXPIREAT', KEYS[2], ARGV[5])
end
local remaining = math.floor((tolerance - (new_tat - now)) / interval)
return {1, remaining, new_tat - now}
"""


@dataclass(frozen=True)
class RateLimitStatus:
    """What an allowed request may report in `RateLimit-*` headers."""

    limit: int
    remaining: int
    reset_seconds: int


class RateLimiter(Protocol):
    """Admits a user's request or raises; records model tokens after the request."""

    async def acquire(self, user_key: str) -> RateLimitStatus | None: ...

    async def charge_tokens(self, user_key: str, tokens: int) -> None: ...

    async def ping(self) -> bool: ...


class RedisRateLimiter:
    """`RateLimiter` on Redis. With `fail_open`, an unreachable Redis admits every request."""

    def __init__(
        self,
        client: Any,
        *,
        requests_per_minute: int,
        burst: int,
        daily_requests: int = 0,
        daily_tokens: int = 0,
        fail_open: bool = False,
        clock: Any = None,
    ) -> None:
        if requests_per_minute < 1 or burst < 1:
            raise ValueError("requests_per_minute and burst must be at least 1")
        self._client = client
        self._script = client.register_script(ACQUIRE_SCRIPT)
        self._interval_ms = 60_000 / requests_per_minute
        self._tolerance_ms = self._interval_ms * burst
        self._burst = burst
        self._daily_requests = daily_requests
        self._daily_tokens = daily_tokens
        self._fail_open = fail_open
        self._clock = clock or (lambda: datetime.now(UTC))

    def _day(self) -> tuple[str, int, int]:
        """Today's key part, the quota keys' expiry, and the seconds to the next UTC day."""
        now: datetime = self._clock()
        midnight = datetime(now.year, now.month, now.day, tzinfo=UTC) + timedelta(days=1)
        expiry = int(midnight.timestamp()) + QUOTA_KEY_GRACE_SECONDS
        return now.strftime("%Y%m%d"), expiry, max(1, math.ceil((midnight - now).total_seconds()))

    def _keys(self, user_key: str, day: str) -> list[str]:
        return [
            f"{KEY_PREFIX}:rate:{user_key}",
            f"{KEY_PREFIX}:requests:{day}:{user_key}",
            f"{KEY_PREFIX}:tokens:{day}:{user_key}",
        ]

    async def acquire(self, user_key: str) -> RateLimitStatus | None:
        """Admit the request, or raise a rate, quota, or availability error.

        Returns `None` when Redis is unreachable and the limiter fails open.
        """
        day, expiry, until_tomorrow = self._day()
        try:
            status, first, second = await self._script(
                keys=self._keys(user_key, day),
                args=[
                    self._interval_ms,
                    self._tolerance_ms,
                    self._daily_requests,
                    self._daily_tokens,
                    expiry,
                ],
            )
        except RedisError as exc:
            self._unavailable(exc)
            return None
        status = int(status)
        if status == 0:
            retry_after = max(1, math.ceil(int(first) / 1000))
            audit("rate_limited", retry_after_s=retry_after)
            raise RateLimitExceededError(retry_after, limit=self._burst)
        if status in (2, 3):
            quota = "requests" if status == 2 else "tokens"
            audit("quota_exceeded", quota=quota)
            raise QuotaExceededError(until_tomorrow, quota=quota)
        return RateLimitStatus(
            limit=self._burst,
            remaining=max(0, int(first)),
            reset_seconds=max(0, math.ceil(int(second) / 1000)),
        )

    async def charge_tokens(self, user_key: str, tokens: int) -> None:
        """Add a finished request's model tokens to today's quota count (never raises)."""
        if self._daily_tokens <= 0 or tokens <= 0:
            return
        day, expiry, _ = self._day()
        key = self._keys(user_key, day)[2]
        try:
            pipeline = self._client.pipeline(transaction=True)
            pipeline.incrby(key, tokens)
            pipeline.expireat(key, expiry)
            await pipeline.execute()
        except RedisError:
            logger.warning("token quota not charged: rate-limit store unavailable")

    async def ping(self) -> bool:
        try:
            return bool(await self._client.ping())
        except RedisError:
            return False

    async def close(self) -> None:
        await self._client.aclose()

    def _unavailable(self, exc: RedisError) -> None:
        audit("rate_limiter_unavailable", fail_open=self._fail_open, error=type(exc).__name__)
        if self._fail_open:
            return None
        raise RateLimiterUnavailableError from exc
