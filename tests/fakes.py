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
from insightflow.providers.rerank import RerankResult
from insightflow.rag.corrective import GRADER_SYSTEM_PROMPT
from insightflow.rag.models import Chunk, RetrievalQuery, RetrievedChunk
from insightflow.rag.query_expansion import DECOMPOSE_SYSTEM_PROMPT, MULTI_QUERY_SYSTEM_PROMPT
from insightflow.rag.refinement import COMPRESSOR_SYSTEM_PROMPT
from insightflow.tools.models import ToolErrorCode, ToolResult, ToolSpec

ROLE_BY_SYSTEM_PROMPT = {
    AGENT_SYSTEM_PROMPT: "agent",
    GROUNDED_SYSTEM_PROMPT: "answer",
    DIRECT_SYSTEM_PROMPT: "direct",
    JUDGE_SYSTEM_PROMPT: "judge",
    MULTI_QUERY_SYSTEM_PROMPT: "expand",
    DECOMPOSE_SYSTEM_PROMPT: "expand",
    GRADER_SYSTEM_PROMPT: "grade",
    COMPRESSOR_SYSTEM_PROMPT: "compress",
}
DEFAULT_REPLIES: dict[str, str] = {
    # An unparseable agent reply makes the agent search the question as written once and then
    # answer, so a test that scripts nothing runs: agent, act, agent, answer, verify, finalize.
    "agent": "not json",
    "answer": "Grounded answer [1].",
    "direct": "Hello! Ask me about the indexed documents.",
    "judge": '{"grounded": true, "unsupported_claims": []}',
    # The retrieval roles default to replies that change nothing.
    "expand": '{"queries": []}',
    "grade": "not json",
    "compress": "not json",
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
    """Returns fixed candidates, or raises one error; records every query.

    `by_query` maps a query text to its own candidates; other queries get `candidates`.
    """

    def __init__(
        self,
        candidates: list[RetrievedChunk] | None = None,
        error: Exception | None = None,
        *,
        by_query: Mapping[str, list[RetrievedChunk]] | None = None,
    ) -> None:
        self.candidates = candidates if candidates is not None else [make_candidate()]
        self.error = error
        self.by_query = dict(by_query or {})
        self.queries: list[RetrievalQuery] = []

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return self.by_query.get(query.text, self.candidates)


class FakeRerankProvider:
    """Scores a document by how many of `keywords` it contains; records every call."""

    def __init__(self, *keywords: str) -> None:
        self.keywords = keywords
        self.calls: list[tuple[str, list[str], int]] = []

    async def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int
    ) -> list[RerankResult]:
        self.calls.append((query, list(documents), top_n))
        scored = [
            RerankResult(
                index=index,
                score=float(sum(document.lower().count(word) for word in self.keywords)),
            )
            for index, document in enumerate(documents)
        ]
        return sorted(scored, key=lambda result: result.score, reverse=True)[:top_n]


def make_candidate(
    chunk_id: str = "chunk-1",
    content: str = "InsightFlow is an API-first RAG platform.",
    *,
    source: str = "PDF/overview.pdf",
    rank: int = 1,
    **chunk_fields: Any,
) -> RetrievedChunk:
    chunk = Chunk.model_validate(
        {
            "chunk_id": chunk_id,
            "document_id": "doc-1",
            "document_version": "v1",
            "chunk_index": rank - 1,
            "content": content,
            "embedding_text": content,
            "page_start": 2,
            "page_end": 2,
            "token_count": len(content.split()),
            "content_type": "paragraph",
            "content_checksum": f"checksum-{chunk_id}",
            "chunker_name": "structure_recursive",
            "chunker_version": "1.0.0",
            **chunk_fields,
        }
    )
    return RetrievedChunk(
        chunk=chunk,
        source_identifier=source,
        source_name=source.rsplit("/", 1)[-1],
        score=0.82,
        rank=rank,
    )


KEYWORDS = ("river", "mountain", "desert")
# Dense search prefers `rivers.txt` for "river CS303" (it counts the letters "river" twice), but
# that file has neither of the query's words. Only the lexical ranking finds `cs303.txt`.
HYBRID_DOCUMENTS = {
    "rivers.txt": "Rivers and riverbanks.",
    "cs303.txt": "Course CS303 has a river and desert trip.",
    "cs101.txt": "Course CS101 has a mountain and desert trip.",
    "peak.txt": "The mountain is tall.",
}
HYBRID_QUERY = "river CS303"


class KeywordEmbeddingProvider:
    """Embeds text as counts of the keyword letters, so dense similarity is predictable."""

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(text.lower().count(word)) + 0.01 for word in KEYWORDS] for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]

    def count_tokens(self, text: str) -> int:
        return max(1, len(text.split()))


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
    `list_error` makes every listing raise; `call_error` makes every call fail with that code.
    """

    def __init__(
        self,
        server_name: str = "fake",
        tools: Sequence[str] = ("search",),
        *,
        replies: Mapping[str, str] | None = None,
        list_error: ToolError | None = None,
        call_error: ToolErrorCode | None = None,
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
        self.call_error = call_error
        self.list_calls = 0
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def server_name(self) -> str:
        return self._server_name

    async def list_tools(self) -> Sequence[ToolSpec]:
        self.list_calls += 1
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
            status="ok" if self.call_error is None else "error",
            error=self.call_error,
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
