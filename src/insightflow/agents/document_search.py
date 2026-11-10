"""The `search_documents` tool: dense retrieval offered to the agent like any other tool.

It runs in-process instead of through MCP, because its result carries the retrieved passages
with their provenance, not only text. Storage and embedding failures are not tool failures: they
raise, as in Stage 3, and the API maps them to error responses.
"""

import time
import uuid
from datetime import UTC, datetime
from typing import Any

from insightflow.agents.models import SEARCH_TOOL
from insightflow.rag.models import ContextBudget, RetrievalQuery, RetrievedChunk
from insightflow.rag.protocols import ContextAssembler, Retriever
from insightflow.tools.models import ToolResult, ToolSpec
from insightflow.tools.validation import argument_problem

LOCAL_SERVER = "insightflow"
MAX_QUERY_CHARS = 500

SEARCH_SPEC = ToolSpec(
    name=SEARCH_TOOL,
    description=(
        "Search the indexed documents and return numbered passages with their sources. Use it "
        "for any fact that the documents can hold. Write a specific, self-contained query; "
        "search again with other words when the passages are not enough."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_QUERY_CHARS,
                "description": "What to look for, in the language of the documents.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
)


class DocumentSearch:
    """Retrieve candidates and keep the passages that fit the context budget."""

    spec = SEARCH_SPEC

    def __init__(
        self,
        retriever: Retriever,
        assembler: ContextAssembler,
        *,
        top_k: int,
        budget: ContextBudget,
    ) -> None:
        self._retriever = retriever
        self._assembler = assembler
        self._top_k = top_k
        self._budget = budget

    async def search(self, arguments: dict[str, Any]) -> tuple[ToolResult, list[RetrievedChunk]]:
        """Run one search; invalid arguments give an error result and no passages."""
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

        problem = argument_problem(SEARCH_SPEC.input_schema, arguments)
        if problem is None and not str(arguments["query"]).strip():
            problem = "Invalid arguments: query: the query is blank"
        if problem is not None:
            return result(problem, error=True), []

        query = str(arguments["query"]).strip()
        candidates = await self._retriever.retrieve(RetrievalQuery(text=query, top_k=self._top_k))
        context = self._assembler.assemble(query, candidates, self._budget)
        return result(f"{len(context.chunks)} passages"), list(context.chunks)
