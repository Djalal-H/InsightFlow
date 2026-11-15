"""The Redis client for rate limits and quotas.

Redis holds only counters that expire (see `security/rate_limit.py`); losing them resets the
limits and needs no backup.
"""

from redis.asyncio import Redis

CONNECT_TIMEOUT_SECONDS = 2.0
SOCKET_TIMEOUT_SECONDS = 2.0


def create_redis_client(url: str) -> Redis:
    """A pooled async client with short timeouts, so a Redis outage fails fast."""
    client: Redis = Redis.from_url(
        url,
        socket_connect_timeout=CONNECT_TIMEOUT_SECONDS,
        socket_timeout=SOCKET_TIMEOUT_SECONDS,
        health_check_interval=30,
        decode_responses=True,
    )
    return client
