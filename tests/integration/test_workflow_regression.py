"""Regression set for the agent workflow through POST /query, with scripted model replies.

Each case in `evaluation/workflow_cases.jsonl` gives the replies per role (see
`tests/fakes.py:ScriptedChatProvider`), optional settings, tools, and retriever behavior, and the
expected run: HTTP status, node order, model roles called, steps, verification, and usage
counters. A change to routing, fallbacks, or budgets shows up here as a changed run.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from insightflow.api.dependencies import get_tool_set
from insightflow.core.exceptions import StorageError
from insightflow.tools.registry import ToolSet
from tests.fakes import (
    FakeToolClient,
    ScriptedChatProvider,
    StaticRetriever,
    make_candidate,
    read_only_registration,
)
from tests.integration.test_query_api import client_for, make_app

DATASET = Path(__file__).parents[2] / "evaluation" / "workflow_cases.jsonl"
CASES = [
    json.loads(line)
    for line in DATASET.read_text(encoding="utf-8").splitlines()
    if line.strip()
]


def retriever_for(kind: str) -> StaticRetriever:
    if kind == "empty":
        return StaticRetriever([])
    if kind == "storage_error":
        return StaticRetriever(error=StorageError("secret detail"))
    assert kind == "one_passage", kind
    return StaticRetriever([make_candidate()])


async def run_case(case: dict[str, Any]) -> tuple[ScriptedChatProvider, dict[str, Any]]:
    chat = ScriptedChatProvider(**case.get("replies", {}))
    application = make_app(
        chat,
        retriever_for(case.get("retriever", "one_passage")),  # type: ignore[arg-type]
        **case.get("settings", {}),
    )
    tools = ToolSet()
    if case.get("tools"):
        client = FakeToolClient("fake", case["tools"], call_error=case.get("tool_error"))
        tools = ToolSet([read_only_registration(client)], failure_cooldown_seconds=0)
    application.dependency_overrides[get_tool_set] = lambda: tools
    async with client_for(application) as client:
        response = await client.post("/query", json={"query": "What is the late fee?"})
    # The HTTP status replaces the body's own `status` (completed / approval_required).
    return chat, {**response.json(), "status": response.status_code}


def observed(chat: ScriptedChatProvider, body: dict[str, Any]) -> dict[str, Any]:
    """The parts of a run that the regression set fixes."""
    if body["status"] != 200:
        return {"status": body["status"], "error": body["error"]["code"], "roles": chat.roles()}
    metadata, usage = body["metadata"], body["metadata"]["usage"]
    return {
        "status": 200,
        "nodes": metadata["nodes"],
        "roles": chat.roles(),
        "attempts": metadata["attempts"],
        "steps": [[step["tool"], step["status"], step["error"]] for step in metadata["steps"]],
        "verification": [body["verification"]["status"], body["verification"]["method"]],
        "sources": len(body["sources"]),
        "usage": {
            "chat_calls": usage["chat"]["calls"],
            "retrieval_calls": usage["retrieval_calls"],
            "retrieved_passages": usage["retrieved_passages"],
            "tool_calls": usage["tool_calls"],
            "tool_errors": usage["tool_errors"],
            "retries": usage["retries"],
        },
    }


def test_case_ids_are_unique() -> None:
    assert len({case["id"] for case in CASES}) == len(CASES) > 0


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
async def test_workflow_run_matches_the_regression_case(case: dict[str, Any]) -> None:
    chat, body = await run_case(case)
    assert observed(chat, body) == case["expected"]
