"""Shared test doubles for the agent workflow."""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from insightflow.agents.nodes.agent import AGENT_SYSTEM_PROMPT
from insightflow.agents.nodes.answer import DIRECT_SYSTEM_PROMPT, GROUNDED_SYSTEM_PROMPT
from insightflow.agents.nodes.verify import JUDGE_SYSTEM_PROMPT
from insightflow.core.exceptions import ToolError
from insightflow.providers.llm import ChatMessage
from insightflow.rag.models import Chunk, RetrievalQuery, RetrievedChunk
from insightflow.tools.models import ToolResult, ToolSpec

ROLE_BY_SYSTEM_PROMPT = {
    AGENT_SYSTEM_PROMPT: "agent",
    GROUNDED_SYSTEM_PROMPT: "answer",
    DIRECT_SYSTEM_PROMPT: "direct",
    JUDGE_SYSTEM_PROMPT: "judge",
}
DEFAULT_REPLIES: dict[str, str] = {
    # An unparseable agent reply makes the agent search the question as written once and then
    # answer, so a test that scripts nothing runs: agent, act, agent, answer, verify, finalize.
    "agent": "not json",
    "answer": "Grounded answer [1].",
    "direct": "Hello! Ask me about the indexed documents.",
    "judge": '{"grounded": true, "unsupported_claims": []}',
}

Reply = str | Exception


class ScriptedChatProvider:
    """Replies to each workflow prompt by its role and records every request.

    A role's replies are used in order; the last one repeats. An exception reply is raised.
    """

    def __init__(self, **replies: Reply | list[Reply]) -> None:
        self.replies: dict[str, list[Reply]] = {
            role: [reply] for role, reply in DEFAULT_REPLIES.items()
        }
        for role, reply in replies.items():
            if role not in DEFAULT_REPLIES:
                raise ValueError(f"unknown role {role}")
            self.replies[role] = reply if isinstance(reply, list) else [reply]
        self.calls: list[tuple[str, list[ChatMessage]]] = []
        self.streamed: list[str] = []

    def calls_for(self, role: str) -> list[list[ChatMessage]]:
        return [messages for called_role, messages in self.calls if called_role == role]

    def roles(self) -> list[str]:
        return [role for role, _ in self.calls]

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        role = ROLE_BY_SYSTEM_PROMPT[messages[0]["content"]]
        self.calls.append((role, list(messages)))
        queue = self.replies[role]
        reply = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        text = await self.complete(messages)
        self.streamed.append(self.calls[-1][0])
        for piece in re.findall(r"\S+\s*", text):
            yield piece


class StaticRetriever:
    """Returns fixed candidates, or raises one error; records every query."""

    def __init__(
        self,
        candidates: list[RetrievedChunk] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.candidates = candidates if candidates is not None else [make_candidate()]
        self.error = error
        self.queries: list[RetrievalQuery] = []

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return self.candidates


def make_candidate(
    chunk_id: str = "chunk-1",
    content: str = "InsightFlow is an API-first RAG platform.",
    *,
    source: str = "PDF/overview.pdf",
    rank: int = 1,
) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        document_version="v1",
        chunk_index=rank - 1,
        content=content,
        embedding_text=content,
        page_start=2,
        page_end=2,
        token_count=len(content.split()),
        content_type="paragraph",
        content_checksum=f"checksum-{chunk_id}",
        chunker_name="structure_recursive",
        chunker_version="1.0.0",
    )
    return RetrievedChunk(
        chunk=chunk,
        source_identifier=source,
        source_name=source.rsplit("/", 1)[-1],
        score=0.82,
        rank=rank,
    )


def word_count(text: str) -> int:
    return len(text.split())


TAVILY_TEXT = """Detailed Results:

Title: FastAPI release notes
URL: https://fastapi.tiangolo.com/release-notes/
Content: FastAPI 0.130 adds native SSE.

Title: Pydantic changelog
URL: https://docs.pydantic.dev/latest/changelog/
Content: Pydantic 2.13 was released.
It improves validation speed."""


class FakeToolClient:
    """A `ToolClient` with fixed tools and replies; records every call.

    `replies` maps a tool name to its output text. Without a reply, a tool echoes its arguments.
    """

    def __init__(
        self,
        server_name: str = "fake",
        tools: Sequence[str] = ("search",),
        *,
        replies: Mapping[str, str] | None = None,
        list_error: ToolError | None = None,
    ) -> None:
        self._server_name = server_name
        self.specs = [
            ToolSpec(
                name=name,
                description=f"The {name} tool.",
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            )
            for name in tools
        ]
        self.replies = dict(replies or {})
        self.list_error = list_error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def server_name(self) -> str:
        return self._server_name

    async def list_tools(self) -> Sequence[ToolSpec]:
        if self.list_error is not None:
            raise self.list_error
        return self.specs

    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> ToolResult:
        self.calls.append((name, dict(arguments)))
        return ToolResult(
            call_id=uuid.uuid4().hex,
            server=self._server_name,
            tool=name,
            arguments=dict(arguments),
            status="ok",
            output=self.replies.get(name, repr(sorted(arguments.items()))),
            started_at=datetime.now(UTC),
            duration_ms=1.0,
        )


def action(tool: str, **arguments: Any) -> str:
    """An agent reply that calls `tool`."""
    return json.dumps({"thought": f"Use {tool}.", "action": tool, "arguments": arguments})


def search(query: str) -> str:
    return action("search_documents", query=query)


ANSWER = '{"thought": "Enough.", "action": "answer"}'
