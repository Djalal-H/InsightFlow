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
- `core`: configuration, logging, and shared errors

Document parsers, MCP integrations, authentication, observability services, and full application
containerization are intentionally deferred until their milestones begin.

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

## Grounded query workflow

`POST /query` runs a LangGraph workflow with conditional edges:

```text
route ─┬─ direct ────────────────────────→ answer → finalize
       └─ retrieve → retrieve → answer → verify ─┬─ finalize
                        ▲                         │
                        └──── rewrite ◄───────────┘  (failed, attempts left)
```

1. `route` asks the chat model for JSON: `retrieve` or `direct` (greetings, questions about the
   assistant, or questions that the conversation already answers), plus a self-contained version
   of the question that resolves references to earlier turns. Unreadable output falls back to
   `retrieve` with the question as written. `AGENT_ROUTER_ENABLED=false` skips this model call.
2. `retrieve` embeds the retrieval query with `EmbeddingProvider.embed_query`, searches Qdrant for
   `RAG_CANDIDATE_TOP_K` chunks (`DenseRetriever`), and selects passages in rank order until
   `RAG_GENERATION_MAX_CHUNKS` or `RAG_CONTEXT_TOKEN_BUDGET` is reached
   (`BudgetedContextAssembler`). Tokens are counted with the embedding model's tokenizer, which
   approximates the chat model's count.
3. `answer` sends the numbered passages and the last `AGENT_HISTORY_TURNS` turns to the chat model,
   with an instruction to answer only from the passages and cite them as `[1]`. When nothing is
   retrieved, it returns a fixed "could not find" answer without a model call. Direct answers use
   only the conversation.
4. `verify` first checks citations without a model (the answer is not empty, cites at least one
   passage, and cites only passages that exist). If that passes and
   `AGENT_VERIFIER_LLM_ENABLED=true`, an LLM judge returns JSON that says whether every claim is
   supported.
5. `rewrite` runs when verification fails and fewer than `AGENT_MAX_ATTEMPTS` answers were made.
   It asks the model for a new retrieval query, and the workflow retrieves again. When the
   attempts are used up, the last answer is returned with `verification.status = "failed"`.
6. `finalize` adds the turn to the conversation history.

A retrieved question makes at most `3 + 3 × (AGENT_MAX_ATTEMPTS − 1)` chat calls: router,
answer, and judge, plus rewrite, answer, and judge for each retry. With the defaults that is 6.

```json
{
  "answer": "InsightFlow is an API-first RAG platform [1].",
  "sources": [
    {
      "citation": 1,
      "source_identifier": "PDF/overview.pdf",
      "source_name": "overview.pdf",
      "chunk_id": "…",
      "page_start": 2,
      "page_end": 2,
      "score": 0.82
    }
  ],
  "thread_id": "3f1c…",
  "verification": {"status": "passed", "method": "llm", "issues": []},
  "metadata": {
    "route": "retrieve",
    "attempts": 1,
    "retrieval_queries": ["What is InsightFlow?"],
    "nodes": ["route", "retrieve", "answer", "verify", "finalize"],
    "duration_ms": 2140
  }
}
```

`sources` lists every passage given to the model in the last attempt; `citation` matches the
numbers in the answer. `verification.status` is `passed`, `failed`, `unchecked` (the judge reply
could not be read), or `skipped` (direct answers and empty retrieval).
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
| `node` | `{"node": "route", "route": "retrieve"}` after each step; `retrieve` adds `passages`, `verify` adds `status` |
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
