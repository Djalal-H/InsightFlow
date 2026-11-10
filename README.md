# InsightFlow

InsightFlow is an API-first multi-agent orchestration and retrieval-augmented generation
platform. The initial implementation is a Python modular monolith built with FastAPI and
LangGraph. Hosted chat and embedding models are accessed through LiteLLM, while Qdrant stores
document vectors and metadata.

## Prerequisites

- Python 3.12 or 3.13
- Docker with Docker Compose, when running Qdrant locally
- API credentials for the hosted model provider selected through LiteLLM

## Local setup

Create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install runtime dependencies when you are ready to begin development:

```bash
python -m pip install -r requirements.txt
```

For development tools, use `requirements-dev.txt` instead. Copy `.env.example` to `.env` and
set `LITELLM_API_KEY`, `LITELLM_API_BASE`, and the exact hosted model identifier. For Doubleword,
use `https://api.doubleword.ai/v1` as the base and prefix the model name with `openai/`, as
required by LiteLLM's OpenAI-compatible routing.

Make one real provider request from the repository root:

```bash
PYTHONPATH=src python scripts/manual_chat_test.py "Explain retrieval augmented generation in one sentence."
```

Start only the local vector database:

```bash
docker compose up -d qdrant
```

Run the application from the repository root:

```bash
uvicorn insightflow.main:app --app-dir src --reload
```

The API exposes `GET /health/live` for process health and `GET /health/ready` for model
configuration and Qdrant connectivity.

## Development checks

```bash
ruff check src tests
mypy src
pytest
```

These checks are configured in `pyproject.toml`. Model and embedding API calls should be mocked
in unit tests; real credentials belong only in opt-in integration tests. Qdrant-backed unit tests
use the client's in-memory mode. Tests marked `qdrant` need a running server and are skipped
unless `QDRANT_TEST_URL` is set; they use synthetic documents and fake embeddings:

```bash
docker compose up -d qdrant
QDRANT_TEST_URL=http://localhost:6333 pytest -m qdrant
```

## Architecture

- `api`: HTTP routes and transport schemas
- `agents`: LangGraph state, nodes, and graph construction
- `providers`: provider-independent chat and embedding contracts with LiteLLM adapters
- `rag`: ingestion and retrieval domain boundaries
- `storage`: Qdrant access
- `tools`: tool contracts, the MCP client adapter, and InsightFlow's own MCP servers
- `core`: configuration, logging, and shared errors

Authentication, observability services, and full application containerization are intentionally
deferred until their milestones begin.

## Max-Min semantic chunking

The reusable `MaxMinSemanticChunker` groups consecutive English sentence units using hosted
embeddings and the published Max-Min similarity rule. It preserves normalized element, heading,
and page provenance while treating `RAG_CHUNK_MAX_TOKENS` as a hard limit and
`RAG_CHUNK_MIN_TOKENS` as a best-effort merge target. Tables, code, headings, titles, and captions
remain atomic unless they exceed the hard token limit.

Select the strategy and optionally tune its published defaults through the environment:

```dotenv
RAG_CHUNKING_STRATEGY=semantic_max_min
RAG_CHUNK_OVERLAP_TOKENS=0
RAG_MAX_MIN_HARD_THRESHOLD=0.6
RAG_MAX_MIN_SIMILARITY_COEFFICIENT=0.9
RAG_MAX_MIN_INITIALIZATION_CONSTANT=1.5
```

Chunking is asynchronous because it embeds sentence units through `EmbeddingProvider`. The
component is available from `insightflow.rag`; parser-to-Qdrant ingestion orchestration remains
separate Stage 3 work.

## Corpus ingestion

`IngestionPipeline` in `insightflow.rag.ingestion` discovers files under a corpus directory,
selects a parser by file type (`.txt`, `.md`, `.docx`, `.pdf`), chunks with the configured
strategy, embeds `embedding_text` through `EmbeddingProvider`, and writes one Qdrant point per
chunk. Each point stores the chunk fields plus source provenance (`source_identifier`,
`source_name`, page range, heading path).

Ingestion requires `EMBEDDING_PROVIDER`, `LITELLM_EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS`, and
`EMBEDDING_MAX_INPUT_TOKENS`. The collection is created with the named vector
`QDRANT_VECTOR_NAME`; an existing collection whose vector size or distance differs is refused,
not migrated.

```bash
docker compose up -d qdrant
PYTHONPATH=src python -m insightflow.ingest corpus/          # or set RAG_CORPUS_DIR
PYTHONPATH=src python -m insightflow.ingest corpus/ --force  # re-embed unchanged documents
```

The command prints one line per document with status `completed`, `skipped`, or `rejected`.
A document is `skipped` when Qdrant already holds every chunk of the same content version
produced with the same parser, chunker, chunking settings, and embedding model. A changed
document replaces all of its previous points. Ingestion makes billable embedding requests.

For manual inspection with real sentence embeddings, open
`notebooks/inspect_max_min_chunking.ipynb`, select the repository `.venv` kernel, set
`DOCUMENT_PATH`, and run the cells in order. The notebook reads model credentials from `.env`,
prints complete chunk provenance and previews, and performs no Qdrant writes. Its embedding cell
makes billable provider requests; the optional JSON export remains disabled until `OUTPUT_PATH` is
set explicitly. Clear executed notebook outputs before committing to avoid retaining document
content in Git.

## Agent query workflow (ReAct)

`POST /query` runs a LangGraph ReAct loop with a verified answer:

```text
agent ⇄ act          (search_documents, calculate, tavily_search, …)
  └→ answer ─┬─ verify ─┬─ finalize
             │          └─ retry → agent   (failed, attempts left)
             └─ finalize                   (no action: direct reply)
```

1. `agent` sends the question, the last `AGENT_HISTORY_TURNS` turns, the tool catalog with input
   schemas, and every observation so far to the chat model. The model returns one JSON action:
   `{"thought": "…", "action": "<tool>", "arguments": {…}}`, or `{"action": "answer"}` when it has
   enough. A tool can be used at any step, for example: search the documents, then calculate with
   a number from a passage, then search the web.
2. `act` runs the action and records the observation. `search_documents` is a local tool: it
   embeds the query, searches Qdrant for `RAG_CANDIDATE_TOP_K` chunks, and keeps passages within
   `RAG_GENERATION_MAX_CHUNKS` and `RAG_CONTEXT_TOKEN_BUDGET` per search. Other actions go to the
   MCP tools (see [MCP tools](#mcp-tools-stage-5)). A failed call is an observation,
   not an error; storage and embedding failures still stop the request.
3. `answer` writes the answer from the observations and cites sources as `[n]`. Document passages
   and web pages share one numbering per turn, in the order they were first observed. Without
   any action, it replies from the conversation only. When the actions found nothing usable, it
   returns a fixed "could not find" answer without a model call.
4. `verify` checks citations without a model (the answer is not empty, cites at least one source,
   and cites only sources that exist). If that passes and `AGENT_VERIFIER_LLM_ENABLED=true`, an
   LLM judge checks every claim against the observations; a calculated value counts as supported.
   Answers with no sources (direct replies, or tool results only) are `skipped`.
5. `retry` runs when verification fails and fewer than `AGENT_MAX_ATTEMPTS` answers were made. It
   sends the issues back to the agent, which gets a new step budget. When the attempts are used
   up, the last answer is returned with `verification.status = "failed"`.
6. `finalize` adds the turn to the conversation history.

**Bounds.** The loop always ends: the agent answers without a model call after
`AGENT_MAX_STEPS` actions in an attempt (default 4), and a call that repeats an earlier call
(same tool, same arguments, ignoring case) ends the attempt. Unreadable agent output searches the
question as written once, then answers. The routes pass a LangGraph `recursion_limit` computed
from both bounds, because the defaults can need more than LangGraph's limit of 25 steps.

An attempt makes at most `AGENT_MAX_STEPS + 2` chat calls: the agent steps, the answer, and the
judge. With the defaults that is 6 per attempt and 12 per question.

```json
{
  "answer": "15% of the 1200 dollar fee [1] is 180.",
  "sources": [
    {
      "kind": "document",
      "citation": 1,
      "source_identifier": "DOCX/tuition.docx",
      "source_name": "tuition.docx",
      "chunk_id": "…",
      "page_start": null,
      "page_end": null,
      "score": 0.82
    }
  ],
  "thread_id": "3f1c…",
  "verification": {"status": "passed", "method": "llm", "issues": []},
  "metadata": {
    "attempts": 1,
    "retrieval_queries": ["annual tuition fee"],
    "steps": [
      {"attempt": 1, "thought": "Find the fee.", "tool": "search_documents",
       "server": "insightflow", "arguments": {"query": "annual tuition fee"}, "call_id": "…",
       "status": "ok", "error": null, "truncated": false, "sources": [1], "duration_ms": 310.2},
      {"attempt": 1, "thought": "Calculate 15%.", "tool": "calculate", "server": "calculator",
       "arguments": {"expression": "0.15 * 1200"}, "call_id": "…", "status": "ok",
       "error": null, "truncated": false, "sources": [], "duration_ms": 11.8}
    ],
    "nodes": ["agent", "act", "agent", "act", "agent", "answer", "verify", "finalize"],
    "duration_ms": 4120
  }
}
```

`sources` lists every distinct source the agent observed; `citation` matches the numbers in the
answer, and `steps[].sources` shows which step returned each one. `verification.status` is
`passed`, `failed`, `unchecked` (the judge reply could not be read), or `skipped`.
Provider failures return the sanitized `provider_*` errors, and Qdrant failures (including a
missing collection before the first ingestion) return HTTP 503 with code `storage_unavailable`.

### Conversation threads

The workflow state is saved by a LangGraph checkpointer after every step.
`CHECKPOINT_BACKEND=sqlite` (the default) stores it in `CHECKPOINT_SQLITE_PATH`
(`data/checkpoints.sqlite`, ignored by Git), so threads survive a restart; `memory` keeps them
only while the process runs; `none` makes every request stateless (`thread_id` is then `null`).

- Send `thread_id` from an earlier response to continue a conversation; omit it to start a new
  thread. Identifiers have 1–128 characters from `A–Z a–z 0–9 _ -`.
- `GET /threads/{thread_id}` returns the completed turns; `DELETE /threads/{thread_id}` removes
  the thread. Unknown threads return HTTP 404 with code `thread_not_found`.
- Concurrent requests on the same thread are not serialized. Send one request per thread at a
  time.

### Streaming

`POST /query/stream` accepts the same body and runs the same workflow, but returns Server-Sent
Events:

| Event | Data |
| --- | --- |
| `node` | `{"node": "agent", "action": "search_documents"}` after each step; `act` adds `tool` and `status`, `verify` adds `status`, `retry` adds `attempt` |
| `token` | `{"attempt": 1, "text": "…"}`: a piece of the answer |
| `attempt` | `{"attempt": 2}`: a retry started; discard the tokens of earlier attempts |
| `final` | the complete `QueryResponse` |
| `error` | `{"error": {"code": …, "message": …}}`: the same sanitized errors as `POST /query` |

```bash
curl -N localhost:8000/query/stream -H 'content-type: application/json' \
  -d '{"query": "What is InsightFlow?"}'
```

Request validation and configuration errors that occur before the stream starts return normal
HTTP errors. After the stream starts, the status is 200 and a failure is sent as one `error` event.

## MCP tools (Stage 5)

InsightFlow is an MCP client. Two tool servers are available:

| Server | Tool | Setting | Notes |
|---|---|---|---|
| Calculator (in this repository, in-process) | `calculate` | `TOOL_CALCULATOR_ENABLED=true` (default) | No network, no key |
| Tavily (hosted, `TAVILY_MCP_URL`) | `tavily_search` only | `TOOL_WEB_SEARCH_ENABLED=true` and `TAVILY_API_KEY` | Billable; off by default |

The application builds the tool set once at startup, so each server's tool list is fetched once.
The agent sees these tools next to `search_documents`, whose name is reserved: an MCP tool with
that name is ignored. Web pages from Tavily become sources with `"kind": "web"`:

```json
{"kind": "web", "citation": 2, "title": "FastAPI release notes", "url": "https://…"}
```

**Controls.** `insightflow.tools.mcp_client.McpToolClient` implements the `ToolClient` contract
for one MCP server with the official MCP SDK (`mcp==2.3.0`). Each call:

- is refused when the tool is not on the client's allowlist or not offered by the server;
- has its arguments checked against the tool's input schema before it is sent;
- stops after `TOOL_TIMEOUT_SECONDS`;
- has its text output limited to `TOOL_MAX_OUTPUT_CHARS` (`truncated` is set when it is cut);
- returns a result with a call ID, start time, duration, and a stable error code
  (`unknown_tool`, `invalid_arguments`, `tool_failed`, `timeout`, `unavailable`). Failures are
  results, not exceptions, and do not include transport or server exception text.

`insightflow.tools.registry.ToolSet` joins the servers into one catalog. A server whose tool
listing fails, or whose call fails with `timeout` or `unavailable`, is paused for
`TOOL_FAILURE_COOLDOWN_SECONDS` (default 60; 0 turns pauses off). While it is paused, the agent
does not see its tools and its calls fail at once with `unavailable`, so a server that is down
costs one timeout, not one on each agent step. The set is shared by all requests, so the pause
applies to all of them. `tool_failed` and `invalid_arguments` do not pause a server.

For Tavily, `max_results` is fixed to `WEB_SEARCH_MAX_RESULTS`, raw page content and images are
turned off, and the API key is sent as an `Authorization: Bearer` header, not in the URL. Web
content goes to the model without prompt-injection filtering; that control is Stage 8 work.

**Calculator.** `insightflow.tools.calculator_server` parses arithmetic with `ast` and evaluates
only allow-listed operators, functions (`abs`, `round`, `min`, `max`, `sqrt`, `floor`, `ceil`,
`log`, `log10`, `exp`), and constants (`pi`, `e`). It never uses `eval`, and it limits
expression length and result size. Other MCP clients can run it over stdio:

```bash
PYTHONPATH=src python -m insightflow.tools.calculator_server
```

**Opt-in Tavily test.** It makes one billable search and uses its own variable, so that a key in
`.env` never makes the default suite billable:

```bash
TAVILY_TEST_API_KEY=tvly-... pytest -m tavily
```
