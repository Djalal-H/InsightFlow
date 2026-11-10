"""HTTP contract tests for POST /query/stream (Server-Sent Events)."""

import json
from typing import Any

import pytest

from insightflow.core.exceptions import (
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    StorageError,
)
from tests.fakes import ScriptedChatProvider, StaticRetriever
from tests.integration.test_query_api import client_for, make_app, memory_saver

Event = tuple[str, Any]


def parse_events(body: str) -> list[Event]:
    """Parse an SSE body into (event, JSON data) pairs; comments are ignored."""
    events: list[Event] = []
    for block in body.replace("\r\n", "\n").split("\n\n"):
        name = "message"
        data_lines: list[str] = []
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())
        if data_lines:
            events.append((name, json.loads("\n".join(data_lines))))
    return events


async def stream_query(application: Any, payload: dict[str, Any]) -> tuple[int, str, list[Event]]:
    async with client_for(application) as client:
        response = await client.post("/query/stream", json=payload)
    return response.status_code, response.headers["content-type"], parse_events(response.text)


async def test_stream_sends_node_token_and_final_events_in_order() -> None:
    chat = ScriptedChatProvider(answer="InsightFlow is a RAG platform [1].")

    status, content_type, events = await stream_query(
        make_app(chat), {"query": "What is InsightFlow?"}
    )

    assert status == 200
    assert content_type.startswith("text/event-stream")
    names = [name for name, _ in events]
    assert names == [
        "node", "node", "node", "token", "token", "token", "token", "token", "token", "node",
        "node", "node", "final",
    ]
    assert [data for name, data in events if name == "node"] == [
        {"node": "agent", "action": "search_documents"},
        {"node": "act", "tool": "search_documents", "status": "ok"},
        {"node": "agent", "action": "answer"},
        {"node": "answer"},
        {"node": "verify", "status": "passed"},
        {"node": "finalize"},
    ]
    text = "".join(data["text"] for name, data in events if name == "token")
    assert text == "InsightFlow is a RAG platform [1]."
    final = events[-1][1]
    assert final["answer"] == text
    assert final["sources"][0]["chunk_id"] == "chunk-1"
    assert final["metadata"]["nodes"] == [
        "agent", "act", "agent", "answer", "verify", "finalize"
    ]


async def test_stream_announces_retries_so_clients_can_discard_old_tokens() -> None:
    chat = ScriptedChatProvider(answer=["No citation.", "Fixed [1]."])

    _, _, events = await stream_query(
        make_app(chat, agent_max_attempts=2), {"query": "What is InsightFlow?"}
    )

    attempt_index = next(i for i, (name, _) in enumerate(events) if name == "attempt")
    assert events[attempt_index] == ("attempt", {"attempt": 2})
    before = [data["text"] for name, data in events[:attempt_index] if name == "token"]
    after = [data for name, data in events[attempt_index:] if name == "token"]
    assert "".join(before) == "No citation."
    assert "".join(data["text"] for data in after) == "Fixed [1]."
    assert {data["attempt"] for data in after} == {2}
    assert events[-1][0] == "final"
    assert events[-1][1]["metadata"]["attempts"] == 2


async def test_stream_keeps_thread_history() -> None:
    application = make_app(ScriptedChatProvider(), checkpointer=memory_saver())

    _, _, first = await stream_query(application, {"query": "first"})
    thread_id = first[-1][1]["thread_id"]
    _, _, second = await stream_query(application, {"query": "second", "thread_id": thread_id})

    assert thread_id
    async with client_for(application) as client:
        thread = await client.get(f"/threads/{thread_id}")
    assert second[-1][1]["thread_id"] == thread_id
    assert [turn["query"] for turn in thread.json()["turns"]] == ["first", "second"]


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (ProviderTimeoutError("secret-detail"), "provider_timeout"),
        (ProviderRateLimitError("secret-detail"), "provider_rate_limited"),
        (ProviderError("secret-detail"), "provider_error"),
    ],
)
async def test_stream_reports_provider_failures_as_an_error_event(
    error: ProviderError, code: str
) -> None:
    chat = ScriptedChatProvider(answer=error)

    status, _, events = await stream_query(make_app(chat), {"query": "q"})

    assert status == 200
    assert events[-1][0] == "error"
    assert events[-1][1]["error"]["code"] == code
    assert "final" not in [name for name, _ in events]
    assert all("secret-detail" not in json.dumps(data) for _, data in events)


async def test_stream_reports_storage_failures_as_an_error_event() -> None:
    retriever = StaticRetriever(error=StorageError("secret-detail"))
    application = make_app(ScriptedChatProvider(), retriever)  # type: ignore[arg-type]

    _, _, events = await stream_query(application, {"query": "q"})

    assert events == [
        ("node", {"node": "agent", "action": "search_documents"}),
        (
            "error",
            {
                "error": {
                    "code": "storage_unavailable",
                    "message": "The document store is unavailable.",
                }
            },
        ),
    ]


async def test_stream_validates_the_request_before_streaming() -> None:
    async with client_for(make_app(ScriptedChatProvider())) as client:
        response = await client.post("/query/stream", json={"query": ""})

    assert response.status_code == 422
