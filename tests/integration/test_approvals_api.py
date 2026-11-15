"""Consequential tools: least privilege, a pause for approval, and the decision endpoint."""

from typing import Any

from insightflow.api.dependencies import get_checkpointer, get_tool_set
from insightflow.tools.registry import ToolRegistration, ToolSet
from tests.fakes import (
    ANSWER,
    AUTH_SETTINGS,
    FakeToolClient,
    ScriptedChatProvider,
    action,
    bearer,
    make_token,
)
from tests.integration.test_query_api import client_for, make_app, memory_saver

SEND = action("send_email", query="hello")
TOOLS_USER = bearer(make_token("alice", "query threads:read tools:consequential"))
PLAIN_USER = bearer(make_token("alice", "query threads:read"))


def approval_app(
    chat: ScriptedChatProvider, *, stored: bool = True, **settings: Any
) -> tuple[Any, FakeToolClient]:
    mailer = FakeToolClient("mailer", ["send_email"], replies={"send_email": "sent"})
    tools = ToolSet([ToolRegistration(mailer)])  # not read-only: consequential
    application = make_app(chat, **settings)
    application.dependency_overrides[get_tool_set] = lambda: tools
    saver = memory_saver() if stored else None
    application.dependency_overrides[get_checkpointer] = lambda: saver
    return application, mailer


def steps(body: dict[str, Any]) -> list[tuple[str, str, str | None]]:
    return [(step["tool"], step["status"], step["error"]) for step in body["metadata"]["steps"]]


async def test_a_consequential_call_waits_for_approval_and_then_runs() -> None:
    application, mailer = approval_app(ScriptedChatProvider(agent=[SEND, ANSWER]))
    async with client_for(application) as client:
        paused = await client.post("/query", json={"query": "Mail Bob"})
        body = paused.json()
        assert (paused.status_code, body["status"], body["answer"]) == (
            200,
            "approval_required",
            "",
        )
        pending = body["pending_approval"]
        assert (pending["tool"], pending["server"], pending["arguments"]) == (
            "send_email",
            "mailer",
            {"query": "hello"},
        )
        assert mailer.calls == []

        url = f"/threads/{body['thread_id']}/approvals/{pending['approval_id']}"
        done = await client.post(url, json={"decision": "approve"})
        # The approval is used up.
        again = await client.post(url, json={"decision": "approve"})

    assert done.status_code == 200
    assert done.json()["status"] == "completed"
    assert steps(done.json()) == [("send_email", "ok", None)]
    assert mailer.calls == [("send_email", {"query": "hello"})]
    assert "approve" in done.json()["metadata"]["nodes"]
    assert (again.status_code, again.json()["error"]["code"]) == (404, "approval_not_found")


async def test_a_rejected_call_never_runs() -> None:
    application, mailer = approval_app(ScriptedChatProvider(agent=[SEND, ANSWER]))
    async with client_for(application) as client:
        body = (await client.post("/query", json={"query": "Mail Bob"})).json()
        url = f"/threads/{body['thread_id']}/approvals/{body['pending_approval']['approval_id']}"
        done = await client.post(url, json={"decision": "reject"})
    assert steps(done.json()) == [("send_email", "error", "not_approved")]
    assert mailer.calls == []


async def test_unknown_and_malformed_approval_ids() -> None:
    application, _ = approval_app(ScriptedChatProvider(agent=[SEND, ANSWER]))
    async with client_for(application) as client:
        body = (await client.post("/query", json={"query": "Mail Bob"})).json()
        thread = body["thread_id"]
        wrong = await client.post(
            f"/threads/{thread}/approvals/{'0' * 32}", json={"decision": "approve"}
        )
        malformed = await client.post(
            f"/threads/{thread}/approvals/not-an-id", json={"decision": "approve"}
        )
        no_thread = await client.post(
            f"/threads/missing/approvals/{'0' * 32}", json={"decision": "approve"}
        )
    assert (wrong.status_code, wrong.json()["error"]["code"]) == (404, "approval_not_found")
    assert malformed.status_code == 422
    assert no_thread.status_code == 404


async def test_without_conversation_storage_a_consequential_call_is_refused() -> None:
    application, mailer = approval_app(ScriptedChatProvider(agent=[SEND, ANSWER]), stored=False)
    async with client_for(application) as client:
        body = (await client.post("/query", json={"query": "Mail Bob"})).json()
    assert body["status"] == "completed"
    assert steps(body) == [("send_email", "error", "not_authorized")]
    assert mailer.calls == []


async def test_users_without_the_scope_do_not_see_or_run_consequential_tools() -> None:
    chat = ScriptedChatProvider(agent=[SEND, ANSWER])
    application, mailer = approval_app(chat, **AUTH_SETTINGS)
    async with client_for(application) as client:
        body = (
            await client.post("/query", json={"query": "Mail Bob"}, headers=PLAIN_USER)
        ).json()
    assert "send_email" not in chat.calls_for("agent")[0][-1]["content"]
    # An injected call to the hidden tool is refused in `act` (defense in depth).
    assert steps(body) == [("send_email", "error", "not_authorized")]
    assert mailer.calls == []


async def test_approving_needs_the_scope_and_the_thread_owner() -> None:
    application, mailer = approval_app(ScriptedChatProvider(agent=[SEND, ANSWER]), **AUTH_SETTINGS)
    async with client_for(application) as client:
        body = (
            await client.post("/query", json={"query": "Mail Bob"}, headers=TOOLS_USER)
        ).json()
        url = f"/threads/{body['thread_id']}/approvals/{body['pending_approval']['approval_id']}"
        other = await client.post(
            url,
            json={"decision": "approve"},
            headers=bearer(make_token("bob", "query tools:consequential")),
        )
        no_scope = await client.post(url, json={"decision": "approve"}, headers=PLAIN_USER)
        rejected = await client.post(url, json={"decision": "reject"}, headers=PLAIN_USER)
    assert other.status_code == 404
    assert (no_scope.status_code, no_scope.json()["error"]["code"]) == (403, "forbidden")
    assert rejected.status_code == 200
    assert mailer.calls == []


async def test_a_new_question_drops_the_pending_approval() -> None:
    chat = ScriptedChatProvider(agent=[SEND, ANSWER, ANSWER])
    application, mailer = approval_app(chat)
    async with client_for(application) as client:
        body = (await client.post("/query", json={"query": "Mail Bob"})).json()
        thread = body["thread_id"]
        url = f"/threads/{thread}/approvals/{body['pending_approval']['approval_id']}"
        follow_up = await client.post("/query", json={"query": "Never mind", "thread_id": thread})
        late = await client.post(url, json={"decision": "approve"})
    assert follow_up.json()["status"] == "completed"
    assert late.status_code == 404
    assert mailer.calls == []


async def test_the_stream_reports_the_pending_approval() -> None:
    from tests.integration.test_query_stream_api import stream_query

    application, _ = approval_app(ScriptedChatProvider(agent=[SEND, ANSWER]))
    status, _, events = await stream_query(application, {"query": "Mail Bob"})
    names = [name for name, _ in events]
    assert status == 200
    assert names[-2:] == ["approval_required", "final"]
    assert events[-2][1]["tool"] == "send_email"
    assert events[-1][1]["status"] == "approval_required"
