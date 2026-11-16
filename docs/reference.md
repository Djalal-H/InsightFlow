# InsightFlow reference

Setup, configuration, and behavior details for every layer of InsightFlow. The
[README](../README.md) gives the overview and the architecture.

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

Install the dependencies. `requirements.txt` is the API runtime, `requirements-ingest.txt` adds
the document parsers (docling with a CPU model stack), and `requirements-dev.txt` adds the test,
lint, and type tools to both:

```bash
python -m pip install -r requirements-dev.txt
```

The Docker images and CI install the exact versions in `requirements.lock` and
`requirements-ingest.lock` (see "Dependencies" below). Copy `.env.example` to `.env` and
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

The default suite also runs two regression sets with no credentials:
`evaluation/structured_cases.jsonl` (raw model outputs per workflow role and the expected parse
or fallback) and `evaluation/workflow_cases.jsonl` (scripted agent runs with the expected node
order, steps, verification, and usage counters). Add a case when a hosted model shows a new
output shape or a routing change is intended.

## Architecture

- `api`: HTTP routes and transport schemas
- `agents`: LangGraph state, nodes, and graph construction
- `providers`: provider-independent chat, embedding, and rerank contracts with LiteLLM adapters
- `rag`: ingestion and retrieval domain boundaries
- `evaluation`: labeled retrieval and answer evaluation
- `observability`: the optional Langfuse trace sink
- `storage`: Qdrant access, the Redis client, and backups (`storage/backup.py`)
- `tools`: tool contracts, the MCP client adapter, and InsightFlow's own MCP servers
- `security`: JWT authentication, scopes, rate limits and quotas, and the audit log
- `guardrails`: prompt-injection controls (input screen, spotlighting, answer checks)
- `core`: configuration, logging, run records (`core/observability.py`), and shared errors

## Observability (Stage 7)

Every HTTP request gets a request ID: a valid `X-Request-ID` header (letters, digits, `_.:-`,
at most 128 characters) is kept, any other value is replaced, and the response returns it in
`X-Request-ID`. The ID is on every log line of the request, with the thread and the workflow
node.

```bash
LOG_FORMAT=json uvicorn insightflow.main:app --app-dir src   # one JSON object per line
LOG_LEVEL=DEBUG LOG_FORMAT=json uvicorn ...                  # also one line per node, model
                                                             # call, Qdrant call, and tool call
```

At `INFO`, a request writes an access line (`request`: method, path, status, duration) and a
`query completed` line (model calls and tokens per kind, store and tool calls, attempts,
verification, errors by code). Failures write `request failed` with the stable error code only.
Questions, prompts, answers, and bodies are not logged, and configured secrets and bearer
tokens are masked in every line. Uvicorn's own access log keeps its format (the Docker image
turns it off, because the `request` line has the same data).

`metadata.usage` in the `/query` response (and the stream's `final` event) has the same
counters: model calls, errors, and tokens per kind (`null` when the provider did not report
them; `unreported_usage` counts such calls), retrieval calls and passages, tool calls and
errors, retries, errors by code, and the time per node.

### Langfuse traces (opt-in)

With `LANGFUSE_ENABLED=true` and keys from a Langfuse project (Langfuse Cloud or your own server
in `LANGFUSE_HOST`), each request is sent as one trace after it ends: a span per workflow node
and, inside it, one observation per model call (model and token use), Qdrant call, and tool
call. The thread is the Langfuse session. Only metadata is sent unless
`LANGFUSE_CAPTURE_CONTENT=true`, which also sends the question, the answer, and each model
call's messages and output. `LANGFUSE_SAMPLE_RATE` keeps a share of the requests. A Langfuse
failure is logged and never fails a request.

```bash
LANGFUSE_TEST_HOST=https://cloud.langfuse.com LANGFUSE_TEST_PUBLIC_KEY=pk-lf-... \
  LANGFUSE_TEST_SECRET_KEY=sk-lf-... pytest -m langfuse   # sends one synthetic trace
```

## Security (Stage 8)

All protections are settings (see `.env.example`). `APP_ENV=production` refuses to start
without authentication, an issuer and audience, rate limits, and JSON logs, and turns the API
docs (`/docs`, `/redoc`, `/openapi.json`) off unless `API_DOCS_ENABLED=true`.

### Authentication and scopes

With `AUTH_ENABLED=true`, every route except `/health/*` needs `Authorization: Bearer <JWT>`.
Use an identity provider's JWKS (`AUTH_JWKS_URL`, RS/ES/PS algorithms) in deployments, or a
shared secret (`AUTH_JWT_SECRET`, HS256) for local work. Only the algorithms in
`AUTH_JWT_ALGORITHMS` are accepted (`none` never); `exp` and `sub` are required, and `iss` and
`aud` when they are configured. Scopes come from the `scope`, `scp`, or `permissions` claim:

| Scope | Allows |
| --- | --- |
| `query` | `POST /query`, `POST /query/stream`, and rejecting a paused tool call |
| `threads:read` / `threads:delete` | `GET` / `DELETE /threads/{id}` of the caller's threads |
| `tools:consequential` | Seeing tools that change data or send something, and approving their calls |
| `admin` | Every scope, and threads of any owner (also threads from before Stage 8, which have no owner) |

A thread belongs to the user who created it (a SHA-256 of the token subject is stored); another
user gets 404, as for a thread that does not exist. With `AUTH_ENABLED=false`, one local user has
every scope. Logs carry `user`, a short hash of the subject, never the subject.

```bash
AUTH_ENABLED=true AUTH_JWT_ALGORITHMS=HS256 AUTH_JWT_SECRET=$(openssl rand -hex 32) ...
PYTHONPATH=src python scripts/issue_dev_token.py --sub dev --scope "query threads:read"
```

### Rate limits and quotas

With `RATE_LIMIT_ENABLED=true` and `REDIS_URL`, each user may send
`RATE_LIMIT_REQUESTS_PER_MINUTE` requests per minute on average, with bursts of
`RATE_LIMIT_BURST` (GCRA in one Redis Lua script, on Redis' clock, so several API instances
share the limits). `QUOTA_DAILY_REQUESTS` and `QUOTA_DAILY_TOKENS` limit each user per UTC day;
tokens are counted from the providers' reported usage after each request. Refusals are 429
`rate_limited` or `quota_exceeded` with `Retry-After`; admitted requests get `RateLimit-Limit`,
`RateLimit-Remaining`, and `RateLimit-Reset`. If Redis cannot be reached, requests get 503
`rate_limiter_unavailable`, or pass with `RATE_LIMIT_FAIL_OPEN=true`.

```bash
REDIS_TEST_URL=redis://localhost:6379/0 pytest -m redis   # the Lua script on a real Redis
```

### Prompt-injection guardrails

No single control stops prompt injection (OWASP LLM01), so the workflow has several layers.
They follow the OWASP LLM Prompt Injection Prevention Cheat Sheet, Microsoft's spotlighting
(Hines et al., 2024), and the agent design patterns of Beurer-Kellner et al. (2025):

1. **Least privilege and human approval.** A tool is consequential unless its registration lists
   it as read-only (the calculator and `tavily_search` are read-only); MCP hints are not
   trusted. Only callers with `tools:consequential` see such tools. A call pauses the run
   (LangGraph `interrupt`); the response has `status: "approval_required"` and the exact tool,
   server, and arguments. `POST /threads/{id}/approvals/{approval_id}` with
   `{"decision": "approve" | "reject"}` continues the run. Without conversation storage, or
   without the scope, the call is refused; `act` checks the approval again.
2. **Spotlighting.** Passages, web pages, and tool output go into `<untrusted-NONCE>` blocks with
   a random nonce per prompt, and the system prompts say that such text is data. With
   `GUARD_DATAMARK_WEB=true`, web text is also datamarked.
3. **Input screening.** Invisible and tag characters are removed from the question and from
   untrusted text. A heuristic screen (NFKC copy, typo-tolerant keywords, role markers, decoded
   Base64) reports `query:*`, `web:*`, `document:*`, and `tool:*` signals in
   `metadata.guardrails` and the audit log. Web pages with a signal are removed before any prompt
   (`GUARD_WEB_ACTION=drop`); a question with a signal is answered (`flag`) or refused with 400
   `input_rejected` (`GUARD_INPUT_ACTION=block`). The screen is a signal, not a boundary.
4. **Answer checks.** Each answer prompt has a random canary; an answer that contains it is
   replaced. Markdown images and links to URLs that are not sources of the turn are removed (a
   rendered image can send data out), secret shapes are redacted, and the length is limited. In a
   stream, `answer_replaced` tells the client to drop the tokens of that attempt; the `final`
   answer is authoritative.
5. **Optional safety model.** `GUARD_SAFETY_MODEL` (for example
   `nvidia_nim/meta/llama-guard-4-12b`) classifies the question and the answer. It finds harmful
   content, not injection, and it fails open. Two extra model calls per request.
6. **Limits.** Questions longer than 4,000 characters get 422, bodies over `MAX_REQUEST_BYTES`
   get 413, and the rate limiter bounds repeated attempts.

`evaluation/injection_cases.jsonl` (34 cases: attacks and benign controls) runs in the default
suite (`tests/unit/test_injection_regression.py`). A false positive on a benign case fails it too.

### Other hardening

Credentials are `SecretStr` and can come from files (`SECRETS_DIR`). LiteLLM's message logging
is off, and the loggers that can print bodies at DEBUG stay at WARNING. Responses have
`X-Content-Type-Options: nosniff`; `/query` and `/threads` responses have
`Cache-Control: no-store`. LiteLLM retries transient provider failures
(`PROVIDER_MAX_RETRIES`).

## Docker, CI, and operations

```bash
docker build --target api -t insightflow-api .        # the API (no docling, no torch)
docker build --target ingest -t insightflow-ingest .  # the ingestion CLI
docker compose --profile app up -d --build            # Qdrant, Redis, and the API
```

`.github/workflows/ci.yml` runs on pull requests and on `main`: Ruff, strict mypy, and the
default suite on Python 3.12 and 3.13; the `redis` and `qdrant` tests against service
containers; both image builds with a smoke test of the API image; and `pip-audit` on the API
lock. Dependabot proposes updates weekly.

Backups, restores, re-indexing with a collection alias, thread retention, and load tests are in
[`operations.md`](operations.md):

```bash
PYTHONPATH=src python -m insightflow.ops backup --out backups/<stamp>
PYTHONPATH=src python -m insightflow.ops restore --from backups/<stamp> [--checkpoints]
PYTHONPATH=src python -m insightflow.ops prune-threads --dry-run
PYTHONPATH=src python scripts/load_test.py --in-process --clients 20 --requests 400
```

### Dependencies

Direct dependencies are pinned in the `requirements*.txt` files. After a change, regenerate the
locks (uv 0.7 or later):

```bash
uv pip compile --universal --python-version 3.12 requirements.txt -o requirements.lock
uv pip compile --universal --python-version 3.12 --torch-backend cpu --emit-index-url \
  requirements-ingest.txt -o requirements-ingest.lock
uv pip install --torch-backend cpu -r requirements-dev.txt -c requirements-ingest.lock
```

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
component is available from `insightflow.rag`.

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

Some embedding models are asymmetric: they embed a question and a passage differently and need
an `input_type` for each. For such a model, set `EMBEDDING_QUERY_INPUT_TYPE` and
`EMBEDDING_DOCUMENT_INPUT_TYPE` (NVIDIA NIM retrieval models: `query` and `passage`; Cohere:
`search_query` and `search_document`). Without them the dense ranking is almost random: on the
evaluation set, the hit rate was 0.500 without the types and 0.979 with them. Leave both empty
for symmetric models such as OpenAI's. A change of the document type embeds every document again
at the next ingestion.

```bash
docker compose up -d qdrant
PYTHONPATH=src python -m insightflow.ingest corpus/          # or set RAG_CORPUS_DIR
PYTHONPATH=src python -m insightflow.ingest corpus/ --force  # re-embed unchanged documents
```

The command prints one line per document with status `completed`, `skipped`, or `rejected`.
A document is `skipped` when Qdrant already holds every chunk of the same content version
produced with the same parser, chunker, chunking settings, and embedding model. A changed
document replaces all of its previous points. Ingestion makes billable embedding requests.

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
  the thread. Unknown threads, and threads of other users, return HTTP 404 with code
  `thread_not_found`.
- `python -m insightflow.ops prune-threads` deletes threads older than `THREAD_RETENTION_DAYS`
  (see [`operations.md`](operations.md)).

Prompts show only the last `AGENT_HISTORY_TURNS` turns. With `MEMORY_ENABLED=true`, each completed
turn of a thread is also embedded and stored in the Qdrant collection `MEMORY_COLLECTION`. For a
new question, a `recall` node runs first: it finds the `MEMORY_TOP_K` older turns (outside that
window) that are most similar to the question, and the prompts show them before the recent turns.
`metadata.recalled_turns` reports how many were added, and `metadata.nodes` starts with `recall`.
`DELETE /threads/{thread_id}` also deletes the remembered turns. Memory is an aid: when its
embedding or storage call fails, a warning is logged and the request continues without it.

### Streaming

`POST /query/stream` accepts the same body and runs the same workflow, but returns Server-Sent
Events:

| Event | Data |
| --- | --- |
| `node` | `{"node": "agent", "action": "search_documents"}` after each step; `act` adds `tool` and `status`, `verify` adds `status`, `retry` adds `attempt`, `recall` adds `recalled_turns`, `approve` adds `status` |
| `token` | `{"attempt": 1, "text": "…"}`: a piece of the answer |
| `attempt` | `{"attempt": 2}`: a retry started; discard the tokens of earlier attempts |
| `answer_replaced` | `{"attempt": 1}`: a guardrail changed the answer; discard the tokens of this attempt |
| `approval_required` | the `pending_approval` of the response: a consequential tool call waits for a decision |
| `final` | the complete `QueryResponse` (its `answer` is authoritative) |
| `error` | `{"error": {"code": …, "message": …}}`: the same sanitized errors as `POST /query` |

```bash
curl -N localhost:8000/query/stream -H 'content-type: application/json' \
  -d '{"query": "What is InsightFlow?"}'
```

Request validation, authentication, rate-limit, and configuration errors that occur before the
stream starts return normal HTTP errors. After the stream starts, the status is 200 and a
failure (also a thread of another user, or a refused question) is sent as one `error` event.

## Retrieval quality (Stage 6)

Plain dense retrieval is the default. Each technique below is an option that is off until you
turn it on, so that you can compare it on the labeled evaluation set before you use it.

| Technique | Setting | What it does | Extra cost per search |
| --- | --- | --- | --- |
| Hybrid dense/sparse | `RAG_RETRIEVAL_MODE=hybrid` | Fuses the dense ranking with a lexical BM25 ranking (reciprocal rank fusion in Qdrant). Finds exact terms such as course codes and names. | None |
| Query rewriting | `RAG_QUERY_EXPANSION=multi_query` | A chat model writes other wordings; all are searched concurrently and fused. | 1 chat call, more embeddings |
| Multi-hop (independent facts) | `RAG_QUERY_EXPANSION=decompose` | A chat model writes one query per fact that the question needs. | 1 chat call, more embeddings |
| Multi-hop (dependent facts) | `AGENT_MULTI_HOP_ENABLED=true` | Adds a planning hint to the agent: one fact per search step, the next query uses what was found. | None |
| Reranking | `LITELLM_RERANK_MODEL=<model>` | A hosted reranker orders the candidates; the best `RAG_RERANK_TOP_N` stay. | 1 rerank call |
| Corrective retrieval | `RAG_CORRECTIVE_RETRIEVAL_ENABLED=true` | A chat model drops irrelevant passages. When none helps, it writes a new query, which is searched once. No relevant passage gives an empty result, so the agent searches differently or uses another tool. | 1–2 chat calls |
| Parent context | `RAG_NEIGHBORS_PER_HIT=1` | Adds the neighboring chunks of each hit to its passage (through the stored `previous_chunk_id`/`next_chunk_id` links). The citation stays the hit. | 1 Qdrant read |
| Context compression | `RAG_CONTEXT_COMPRESSION_ENABLED=true` | A chat model selects the sentences that help; the passage keeps only those, word for word, with `[...]` for omissions. | 1 chat call |
| Metadata filters | request `filters`; `RAG_SEARCH_FILTERS_ENABLED=true` | Limits search by file type and source path (see below). | None |
| Conversation memory | `MEMORY_ENABLED=true` | Recalls older turns of the thread by similarity (see [Conversation threads](#conversation-threads)). | 2 embedding calls per turn |

The steps run in this order: search (dense or hybrid) → query expansion and fusion → reranking →
corrective grading → neighbor expansion → compression → context budget. `rag/composition.py`
builds this chain from `RetrievalConfig`; the agent sees only the `Retriever` protocol.

Notes:

- **No local models.** The sparse vectors are computed in Python (`rag/sparse.py`: hashed terms
  with BM25 weights; Qdrant applies IDF). Reranking and the chat steps use hosted APIs.
- **Hybrid needs a hybrid index.** Set `RAG_RETRIEVAL_MODE=hybrid`, use a new
  `QDRANT_COLLECTION`, and run the ingestion again. A dense-only collection is refused. In hybrid
  mode, `score` in the response is a fusion score, not a similarity. After reranking it is the
  reranker's relevance score.
- **Safe fallbacks.** When a chat step returns unreadable JSON, the step changes nothing: the
  original query is searched, the passages stay, and nothing is compressed. Provider failures
  still return the sanitized `provider_*` errors.
- **Cost.** With every chat step on, one `search_documents` call makes up to five more chat calls,
  and the agent can search `AGENT_MAX_STEPS` times per attempt.

### Metadata filters

A request can limit every document search of the turn:

```bash
curl -s localhost:8000/query -H 'content-type: application/json' -d '{
  "query": "What are the quiet hours?",
  "filters": {"file_types": ["pdf"], "sources": ["PDF/student_handbook.pdf"]}
}'
```

`file_types` accepts `pdf`, `docx`, `md`, and `txt`; `sources` takes corpus-relative paths as
`source_identifier` reports them. Web search ignores the filters. With
`RAG_SEARCH_FILTERS_ENABLED=true`, the agent also gets `file_types` and `sources` as optional
arguments of `search_documents`. Its filters can only narrow the request's filters.

### Evaluation

`evaluation/retrieval_cases.jsonl` holds 48 labeled questions about the sample corpus. Each case
names the source document and a text that a relevant passage contains, so the labels stay valid
when chunking changes. The tags are `fact`, `keyword`, `paraphrase`, `table`, `split` (the answer
is in a chunk without the words of the question), `prose`, and `multi_hop` (two labeled passages).

```bash
# Index the corpus with sparse vectors, in a collection of its own (billable embeddings).
RAG_RETRIEVAL_MODE=hybrid QDRANT_COLLECTION=insightflow_eval \
  PYTHONPATH=src python -m insightflow.ingest corpus/

# Compare the techniques on that index (billable: embeddings, chat, and rerank calls).
QDRANT_COLLECTION=insightflow_eval PYTHONPATH=src python -m insightflow.evaluate
QDRANT_COLLECTION=insightflow_eval PYTHONPATH=src python -m insightflow.evaluate \
  --variants dense,hybrid,hybrid+rerank+neighbors --tag multi_hop --output scores.json
```

The command scores the passages that enter the model's context (after the chunk and token
budget): `hit` (a labeled passage is present), `recall` (the share of labeled passages present),
`mrr`, the number of passages, the context tokens, and the retrieval time. A variant is plain
dense retrieval plus the named options (`hybrid`, `multi_query`, `decompose`, `rerank`,
`corrective`, `neighbors`, `compress`); `configured` is the `.env` setup. The command reads the
collection and writes nothing to it. It does not measure answer quality. Each variant also
reports its model calls and tokens. `--baseline scores.json` compares a run with an earlier
`--output` file and exits with code 3 when hit, recall, or mrr fell more than `--tolerance`
(default 0.02).

#### Answer quality

`evaluation/answer_cases.jsonl` holds 28 questions: 25 with the facts that a correct answer
states (with accepted phrasings, checked against the corpus text) and 3 that the corpus cannot
answer. `insightflow.evaluate_answers` runs the full workflow for each case (as `POST /query`,
no thread, no tools unless `--with-tools`) and one judge call, and reports fact recall (by
phrasing and by the judge), unsupported claims, correct refusals, citation validity, source
coverage, verification, retries, model calls, tokens, and latency.

```bash
PYTHONPATH=src python -m insightflow.evaluate_answers --dry-run          # estimate, no request
PYTHONPATH=src python -m insightflow.evaluate_answers --max-chat-calls 150 --output answers.json
PYTHONPATH=src python -m insightflow.evaluate_answers --variants configured,dense+rerank --max-cases 10
```

The run is billable. It stops when `--max-chat-calls` (default 200) or `--max-tokens` (default
400,000) is used up, and prints the partial report. An unreadable judge verdict leaves a case
unjudged; it does not count as a pass. With `LANGFUSE_ENABLED`, each case is one trace.

Measured on 2026-11-15 with the `.env` setup (dense retrieval, NVIDIA NIM chat and embedding
models): fact recall 0.960 (judge 0.940), citations
valid 1.000, source coverage 0.960, verification passed 0.893, 167 chat calls for 28 cases. On the retrieval set, `dense+corrective` was the
only option with a gain (hit 1.000, mrr 0.979, 268 context tokens instead of 1147);
`dense+compress` lost passages; `multi_query` and `decompose` gave no gain.

Measured on 2026-11-13 with `nvidia_nim/nvidia/nemotron-3-embed-1b`:

| Variant | hit | recall | mrr | context tokens |
| --- | --- | --- | --- | --- |
| dense, no embedding input types | 0.500 | 0.469 | 0.398 | 1818 |
| dense | 0.979 | 0.979 | 0.915 | 1147 |
| hybrid | 0.979 | 0.958 | 0.924 | 1725 |
| dense+neighbors | 0.979 | 0.979 | 0.915 | 1624 |
| hybrid+neighbors | 0.979 | 0.958 | 0.913 | 3083 |

The first row shows why the embedding input types are important (see
[Corpus ingestion](#corpus-ingestion)). With them, dense retrieval finds 47 of the 48 cases, so
this set can no longer separate the other options, and dense stays the default.

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
pages are screened before they enter a prompt: a page with an injection signal is dropped by
default (`GUARD_WEB_ACTION`), and the others go in spotlight blocks (see Security).

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
