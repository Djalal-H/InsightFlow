"""HTTP contract tests for agent tool use through POST /query and POST /query/stream."""

from types import SimpleNamespace
from typing import Any

import pytest

from insightflow import main
from insightflow.api.dependencies import get_tool_set
from insightflow.core.config import Settings
from insightflow.tools.calculator_server import create_calculator_server
from insightflow.tools.mcp_client import McpToolClient
from insightflow.tools.registry import ToolRegistration, ToolSet
from insightflow.tools.web_search import parse_tavily_sources
from tests.fakes import ANSWER, TAVILY_TEXT, FakeToolClient, ScriptedChatProvider, action, search
from tests.integration.test_query_api import FakeRetriever, client_for, make_app, make_candidate
from tests.integration.test_query_stream_api import stream_query

CALCULATE = action("calculate", expression="0.15 * 1200")


def app_with_tools(
    chat: ScriptedChatProvider,
    tools: ToolSet,
    retriever: FakeRetriever | None = None,
    **settings: Any,
) -> Any:
    application = make_app(chat, retriever or FakeRetriever([make_candidate()]), **settings)
    application.dependency_overrides[get_tool_set] = lambda: tools
    return application


def calculator_tools() -> ToolSet:
    calculator = McpToolClient(create_calculator_server(), server_name="calculator")
    return ToolSet([ToolRegistration(calculator)])


async def test_query_reports_each_step_with_its_tool_call() -> None:
    chat = ScriptedChatProvider(
        agent=[search("tuition fee"), CALCULATE, ANSWER],
        answer="15% of the fee [1] is 180.",
    )

    async with client_for(app_with_tools(chat, calculator_tools())) as client:
        response = await client.post("/query", json={"query": "What is 15% of the fee?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "15% of the fee [1] is 180."
    assert [source["kind"] for source in body["sources"]] == ["document"]
    assert body["verification"]["status"] == "passed"
    metadata = body["metadata"]
    assert metadata["nodes"] == [
        "agent", "act", "agent", "act", "agent", "answer", "verify", "finalize"
    ]
    assert metadata["retrieval_queries"] == ["tuition fee"]
    search_step, calculate_step = metadata["steps"]
    assert (search_step["tool"], search_step["server"], search_step["sources"]) == (
        "search_documents",
        "insightflow",
        [1],
    )
    for key in ("call_id", "duration_ms"):
        calculate_step.pop(key)
    assert calculate_step == {
        "attempt": 1,
        "thought": "Use calculate.",
        "tool": "calculate",
        "server": "calculator",
        "arguments": {"expression": "0.15 * 1200"},
        "status": "ok",
        "error": None,
        "truncated": False,
        "sources": [],
    }


async def test_query_returns_document_and_web_sources_in_one_numbering() -> None:
    web = FakeToolClient("tavily", ["web_search"], replies={"web_search": TAVILY_TEXT})
    tools = ToolSet([ToolRegistration(web, source_parser=parse_tavily_sources)])
    chat = ScriptedChatProvider(
        agent=[search("FastAPI"), action("web_search", query="FastAPI news"), ANSWER],
        answer="FastAPI [1] adds native SSE [2].",
    )

    async with client_for(app_with_tools(chat, tools)) as client:
        response = await client.post("/query", json={"query": "What is new in FastAPI?"})

    body = response.json()
    assert [(source["kind"], source["citation"]) for source in body["sources"]] == [
        ("document", 1),
        ("web", 2),
        ("web", 3),
    ]
    assert body["sources"][1] == {
        "kind": "web",
        "citation": 2,
        "title": "FastAPI release notes",
        "url": "https://fastapi.tiangolo.com/release-notes/",
    }
    assert [step["sources"] for step in body["metadata"]["steps"]] == [[1], [2, 3]]


async def test_query_respects_the_step_budget() -> None:
    plans = [action("calculate", expression=f"{n} + 1") for n in range(9)]
    chat = ScriptedChatProvider(agent=plans)

    application = app_with_tools(chat, calculator_tools(), agent_max_steps=3)
    async with client_for(application) as client:
        response = await client.post("/query", json={"query": "q"})

    assert response.status_code == 200
    assert len(response.json()["metadata"]["steps"]) == 3


async def test_default_bounds_run_within_the_recursion_limit() -> None:
    """Four steps and two attempts need more than LangGraph's default limit of 25."""
    chat = ScriptedChatProvider(
        agent=[search(f"query {n}") for n in range(20)], answer="No citation."
    )

    application = app_with_tools(
        chat, ToolSet(), agent_max_steps=4, agent_max_attempts=2
    )
    async with client_for(application) as client:
        response = await client.post("/query", json={"query": "q"})

    assert response.status_code == 200
    body = response.json()
    assert len(body["metadata"]["steps"]) == 8
    assert body["verification"]["status"] == "failed"


async def test_stream_reports_agent_and_act_events() -> None:
    chat = ScriptedChatProvider(agent=[CALCULATE, ANSWER], answer="180.")

    status, _, events = await stream_query(
        app_with_tools(chat, calculator_tools()), {"query": "What is 15% of 1200?"}
    )

    assert status == 200
    assert [data for name, data in events if name == "node"] == [
        {"node": "agent", "action": "calculate"},
        {"node": "act", "tool": "calculate", "status": "ok"},
        {"node": "agent", "action": "answer"},
        {"node": "answer"},
        {"node": "verify", "status": "skipped"},
        {"node": "finalize"},
    ]
    assert events[-1][0] == "final"
    assert events[-1][1]["metadata"]["steps"][0]["tool"] == "calculate"


async def test_stream_announces_retries() -> None:
    chat = ScriptedChatProvider(
        agent=[search("a"), ANSWER, search("b"), ANSWER],
        answer=["No citation.", "Fixed [1]."],
    )

    _, _, events = await stream_query(
        app_with_tools(chat, ToolSet(), agent_max_attempts=2), {"query": "q"}
    )

    assert ("node", {"node": "retry", "attempt": 2}) in events
    assert ("attempt", {"attempt": 2}) in events


def test_openapi_documents_steps_and_web_sources() -> None:
    schemas = main.create_app().openapi()["components"]["schemas"]

    assert "WebSourceReference" in schemas
    assert "StepReport" in schemas
    assert "route" not in schemas["ExecutionMetadata"]["properties"]
    assert "steps" in schemas["ExecutionMetadata"]["properties"]


async def test_lifespan_builds_the_tool_set(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(
        _env_file=None,
        checkpoint_backend="none",
        tool_calculator_enabled=True,
        tool_web_search_enabled=False,
    )
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    application = main.create_app()

    async with main.lifespan(application):
        tools = application.state.tool_set
        request = SimpleNamespace(app=application)
        assert get_tool_set(request) is tools  # type: ignore[arg-type]
        assert [spec.name for spec in await tools.list_tools()] == ["calculate"]


def test_get_tool_set_is_empty_without_a_lifespan() -> None:
    request = SimpleNamespace(app=main.create_app())

    assert get_tool_set(request).is_empty  # type: ignore[arg-type]
