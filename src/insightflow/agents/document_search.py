"""The `search_documents` tool: dense retrieval offered to the agent like any other tool.

It runs in-process instead of through MCP, because its result carries the retrieved passages
with their provenance, not only text. Storage and embedding failures are not tool failures: they
raise, as in Stage 3, and the API maps them to error responses.

Metadata filters come from two places. The request can set a scope (`file_types`, `sources`)
that limits every search of the turn. With `filters_enabled`, the agent can also pass the same
two filters as tool arguments; they can only narrow the request scope.
"""

import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from insightflow.agents.models import SEARCH_TOOL
from insightflow.rag.ingestion import MIME_TYPES_BY_SUFFIX
from insightflow.rag.models import ContextBudget, RetrievalQuery, RetrievedChunk
from insightflow.rag.protocols import ContextAssembler, Retriever
from insightflow.tools.models import ToolResult, ToolSpec
from insightflow.tools.validation import argument_problem

LOCAL_SERVER = "insightflow"
MAX_QUERY_CHARS = 500

MAX_SOURCE_FILTERS = 20
FILE_TYPES = ("pdf", "docx", "md", "txt")
MIME_TYPES_BY_FILE_TYPE: dict[str, list[str]] = {
    "pdf": [MIME_TYPES_BY_SUFFIX[".pdf"], "application/x-pdf"],
    "docx": [MIME_TYPES_BY_SUFFIX[".docx"]],
    "md": [MIME_TYPES_BY_SUFFIX[".md"]],
    "txt": [MIME_TYPES_BY_SUFFIX[".txt"]],
}
_FILTER_PROPERTIES: dict[str, Any] = {
    "file_types": {
        "type": "array",
        "items": {"type": "string", "enum": list(FILE_TYPES)},
        "minItems": 1,
        "uniqueItems": True,
        "description": "Optional. Search only documents of these file types.",
    },
    "sources": {
        "type": "array",
        "items": {"type": "string", "minLength": 1},
        "minItems": 1,
        "maxItems": MAX_SOURCE_FILTERS,
        "description": (
            "Optional. Search only these documents. Use source paths exactly as earlier "
            "passages show them, for example PDF/student_handbook.pdf."
        ),
    },
}


def search_spec(*, filters: bool = False) -> ToolSpec:
    """The tool definition; with `filters`, the agent can limit a search by metadata."""
    properties: dict[str, Any] = {
        "query": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_QUERY_CHARS,
            "description": "What to look for, in the language of the documents.",
        }
    }
    description = (
        "Search the indexed documents and return numbered passages with their sources. Use it "
        "for any fact that the documents can hold. Write a specific, self-contained query; "
        "search again with other words when the passages are not enough."
    )
    if filters:
        properties.update(_FILTER_PROPERTIES)
        description += (
            " Use file_types or sources only when the question names a document or a kind of "
            "document."
        )
    return ToolSpec(
        name=SEARCH_TOOL,
        description=description,
        input_schema={
            "type": "object",
            "properties": properties,
            "required": ["query"],
            "additionalProperties": False,
        },
    )


SEARCH_SPEC = search_spec()


class DocumentSearch:
    """Retrieve candidates and keep the passages that fit the context budget."""

    def __init__(
        self,
        retriever: Retriever,
        assembler: ContextAssembler,
        *,
        top_k: int,
        budget: ContextBudget,
        filters_enabled: bool = False,
    ) -> None:
        self.spec = search_spec(filters=filters_enabled)
        self._retriever = retriever
        self._assembler = assembler
        self._top_k = top_k
        self._budget = budget

    async def search(
        self,
        arguments: dict[str, Any],
        *,
        scope: Mapping[str, Sequence[str]] | None = None,
    ) -> tuple[ToolResult, list[RetrievedChunk]]:
        """Run one search; invalid arguments give an error result and no passages.

        `scope` is the request's filter. A tool filter with nothing in the scope finds nothing.
        """
        call_id = uuid.uuid4().hex
        started_at = datetime.now(UTC)
        start = time.perf_counter()

        def result(output: str, *, error: bool = False) -> ToolResult:
            return ToolResult(
                call_id=call_id,
                server=LOCAL_SERVER,
                tool=SEARCH_TOOL,
                arguments=arguments,
                status="error" if error else "ok",
                error="invalid_arguments" if error else None,
                output=output,
                started_at=started_at,
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        problem = argument_problem(self.spec.input_schema, arguments)
        if problem is None and not str(arguments["query"]).strip():
            problem = "Invalid arguments: query: the query is blank"
        if problem is not None:
            return result(problem, error=True), []

        query = str(arguments["query"]).strip()
        scope = scope or {}
        file_types = _narrow(scope.get("file_types"), arguments.get("file_types"))
        sources = _narrow(scope.get("sources"), arguments.get("sources"))
        if file_types == [] or sources == []:
            return result("0 passages"), []
        candidates = await self._retriever.retrieve(
            RetrievalQuery(
                text=query,
                top_k=self._top_k,
                source_types=(
                    [mime for name in file_types for mime in MIME_TYPES_BY_FILE_TYPE.get(name, [])]
                    if file_types is not None
                    else None
                ),
                source_identifiers=sources,
            )
        )
        context = self._assembler.assemble(query, candidates, self._budget)
        return result(f"{len(context.chunks)} passages"), list(context.chunks)


def _narrow(scope: Sequence[str] | None, requested: Sequence[str] | None) -> list[str] | None:
    """Combine the request scope with a tool filter: `None` is no filter, `[]` matches nothing."""
    if not scope:
        return list(requested) if requested else None
    if not requested:
        return list(scope)
    return [value for value in requested if value in scope]
