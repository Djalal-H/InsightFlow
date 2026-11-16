"""Load test for POST /query: concurrent clients, latency percentiles, and status counts.

In-process (no network, no cost): the app runs with a scripted chat model that sleeps to
simulate provider latency, a static retriever, and an in-memory Redis for the rate limiter::

    PYTHONPATH=src python scripts/load_test.py --in-process --clients 20 --requests 400
    PYTHONPATH=src python scripts/load_test.py --in-process --rate-limit 600 --burst 5

Against a deployment (billable: every request calls the hosted models)::

    PYTHONPATH=src python scripts/load_test.py --url https://api.example --token "$TOKEN" \
        --clients 5 --requests 50

The in-process mode measures the application's own overhead and its limits under concurrency,
not provider or network latency.
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import Counter
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # for tests.fakes in the in-process mode


def build_in_process_app(args: argparse.Namespace) -> Any:
    from fakeredis import FakeAsyncRedis
    from tests.fakes import AUTH_SETTINGS, ScriptedChatProvider, StaticRetriever, make_candidate

    from insightflow.api.dependencies import (
        get_chat_provider,
        get_checkpointer,
        get_context_assembler,
        get_rate_limiter,
        get_retriever,
    )
    from insightflow.core.config import Settings, get_settings
    from insightflow.main import create_app
    from insightflow.providers.llm import ChatMessage
    from insightflow.rag.retrieval import BudgetedContextAssembler
    from insightflow.security.rate_limit import RedisRateLimiter

    latency = args.latency_ms / 1000

    class SlowProvider(ScriptedChatProvider):
        async def complete(self, messages: Sequence[ChatMessage]) -> str:
            await asyncio.sleep(latency)
            return await super().complete(messages)

        async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
            await asyncio.sleep(latency)
            async for piece in super().stream(messages):
                yield piece

    provider = SlowProvider()
    retriever = StaticRetriever([make_candidate()])
    settings = Settings(
        _env_file=None,
        agent_verifier_llm_enabled=False,
        agent_max_attempts=1,
        **(AUTH_SETTINGS if args.rate_limit else {}),
    )
    application = create_app()
    application.dependency_overrides[get_settings] = lambda: settings
    application.dependency_overrides[get_chat_provider] = lambda: provider
    application.dependency_overrides[get_retriever] = lambda: retriever
    application.dependency_overrides[get_checkpointer] = lambda: None
    application.dependency_overrides[get_context_assembler] = lambda: BudgetedContextAssembler(
        lambda text: len(text.split())
    )
    if args.rate_limit:
        limiter = RedisRateLimiter(
            FakeAsyncRedis(decode_responses=True),
            requests_per_minute=args.rate_limit,
            burst=args.burst,
        )
        application.dependency_overrides[get_rate_limiter] = lambda: limiter
    return application


def client_tokens(args: argparse.Namespace) -> list[str | None]:
    """One token per client: with the in-process limiter, each client is its own user."""
    if args.url:
        return [args.token] * args.clients
    if not args.rate_limit:
        return [None] * args.clients
    from tests.fakes import make_token

    return [make_token(f"load-user-{index}", expires_in=3600) for index in range(args.clients)]


async def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.url:
        transport: httpx.AsyncBaseTransport = httpx.AsyncHTTPTransport(retries=0)
        base_url = args.url.rstrip("/")
    else:
        transport = httpx.ASGITransport(app=build_in_process_app(args))
        base_url = "http://load.test"
    tokens = client_tokens(args)
    queue: asyncio.Queue[int] = asyncio.Queue()
    for index in range(args.requests):
        queue.put_nowait(index)
    latencies: list[float] = []
    statuses: Counter[str] = Counter()

    async def worker(token: str | None) -> None:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        async with httpx.AsyncClient(
            transport=transport, base_url=base_url, timeout=args.timeout
        ) as client:
            while True:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                started = time.perf_counter()
                try:
                    response = await client.post(
                        "/query", json={"query": args.query}, headers=headers
                    )
                    key = str(response.status_code)
                    if response.status_code >= 400:
                        try:
                            key += ":" + response.json()["error"]["code"]
                        except (ValueError, KeyError, TypeError):
                            pass
                except httpx.HTTPError as exc:
                    key = f"transport:{type(exc).__name__}"
                latencies.append((time.perf_counter() - started) * 1000)
                statuses[key] += 1

    started = time.perf_counter()
    await asyncio.gather(*(worker(token) for token in tokens))
    elapsed = time.perf_counter() - started
    ordered = sorted(latencies)

    def percentile(share: float) -> float:
        return round(ordered[min(len(ordered) - 1, int(share * len(ordered)))], 1)

    limited = sum(count for key, count in statuses.items() if key.startswith("429"))
    return {
        "mode": "url" if args.url else "in-process",
        "clients": args.clients,
        "requests": len(latencies),
        "duration_s": round(elapsed, 2),
        "throughput_rps": round(len(latencies) / elapsed, 1) if elapsed else None,
        "latency_ms": {
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "max": round(ordered[-1], 1),
            "mean": round(statistics.fmean(ordered), 1),
        },
        "statuses": dict(sorted(statuses.items())),
        "rate_limited_share": round(limited / len(latencies), 3),
        "simulated_provider_latency_ms": None if args.url else args.latency_ms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Load test POST /query.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--in-process", action="store_true", help="Fakes, no network, no cost.")
    target.add_argument("--url", help="A running deployment (billable).")
    parser.add_argument("--token", help="Bearer token for --url.")
    parser.add_argument("--clients", type=int, default=20)
    parser.add_argument("--requests", type=int, default=400)
    parser.add_argument("--query", default="What is InsightFlow?")
    parser.add_argument("--latency-ms", type=float, default=20.0, help="Simulated, per model call.")
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=0,
        help="In-process: requests per minute per user (turns auth and the limiter on).",
    )
    parser.add_argument("--burst", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    if args.clients < 1 or args.requests < 1:
        parser.error("--clients and --requests must be at least 1")
    print(json.dumps(asyncio.run(run(args)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
