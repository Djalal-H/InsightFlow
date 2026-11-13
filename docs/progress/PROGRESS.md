# InsightFlow Progress

## Current status

- Current stage: Stage 6 complete on `feature/stage6-retrieval-quality` (closed by the author's decision). Every retrieval-quality candidate is an option that is off by default; the embedding `input_type` fix is measured; the chat-backed options, reranking, and memory are verified with mocks only.
- Last updated: 2026-11-13
- Overall state: in progress

## Stage summary

| Stage   | Status      | Evidence                                                                                                    |
| ------- | ----------- | ----------------------------------------------------------------------------------------------------------- |
| Stage 1 | in progress | Repository foundation is present, but tests, runtime dependency pins, and mypy validation remain incomplete |
| Stage 2 | complete    | Query workflow, provider validation and error mapping, and mocked contract coverage verified                |
| Stage 3 | complete | Ingestion, Qdrant persistence, dense retrieval, and grounded answers with sources verified with mocks and on a real Qdrant server; a hosted-model run on `corpus/MD` was reported working by the author (2026-10-29) |
| Stage 4 | in progress | Router, verifier, bounded retries, execution metadata, SQLite checkpointer with threads, and SSE streaming verified with mocks; no hosted-model run yet |
| Stage 5 | complete | MCP client adapter with per-call controls, calculator MCP server, ReAct agent loop with document search as a tool, Tavily web search with web sources, and a tool server cooldown, verified with mocks and in-process servers; the opt-in Tavily test and three hosted-model queries passed (2026-11-10) |
| Stage 6 | complete | All candidates implemented and verified with mocks and on a real Qdrant server. Measured on 48 labeled questions with hosted embeddings: the `input_type` fix raised the dense hit rate from 0.500 to 0.979; hybrid and neighbor expansion gave no gain after it. Closed by the author's decision with the chat-backed options, reranking, and memory not measured on hosted models |
| Stage 7 | not started | —                                                                                                          |
| Stage 8 | not started | —                                                                                                          |

## Work log

### 2026-07-17 — Stage 1 foundation verification

- Stage: Stage 1
- Status: partial
- Scope: Verified the repository and development foundation against the implementation roadmap.
- Changed:
  - `src/insightflow/` — modular API, agent, RAG, provider, storage, and core boundaries are present.
  - `compose.yaml` — Qdrant-only Compose service has persistent storage and a health check.
  - `.env.example`, `README.md`, `docs/Project Overview.md` — configuration, setup, and architecture documentation are present.
  - `requirements.txt`, `requirements-dev.txt`, `pyproject.toml` — dependency and development-tool declarations are present.
  - `.venv/` — dedicated Python 3.13.3 virtual environment is present.
- Validation:
  - `./.venv/bin/ruff check src tests` — passed
  - `./.venv/bin/mypy src` — failed: LangGraph typing errors in `src/insightflow/agents/graph.py`
  - `./.venv/bin/python -m pytest -q` — failed: no tests ran
  - `git status --short` — passed: only pre-existing untracked `.agents/` is present
- Decisions:
  - Stage 1 is not marked complete because runtime dependencies are unpinned and the test suite is empty.
- Remaining:
  - Pin direct runtime dependencies.
  - Add Stage 1 unit/integration coverage, including health and adapter boundaries.
  - Resolve the LangGraph typing errors so mypy passes.
- Commit: `2987fe0`

### 2026-07-24 — Stage 2 minimal hosted-model workflow verification

- Stage: Stage 2
- Status: completed
- Scope: Verified the committed non-streaming query workflow, provider-independent composition, stable provider error mapping, and mocked API/provider coverage.
- Changed:
  - `src/insightflow/api/routes/query.py`, `src/insightflow/api/schemas.py` — exposes the typed `POST /query` request/response contract and documented provider-failure responses.
  - `src/insightflow/agents/graph.py`, `src/insightflow/agents/nodes/answer.py` — executes the minimal LangGraph answer node through the chat-provider protocol.
  - `src/insightflow/api/dependencies.py`, `src/insightflow/providers/llm.py` — composes LiteLLM behind the provider contract and validates/mapping provider failures.
  - `src/insightflow/api/errors.py`, `src/insightflow/main.py` — returns stable, sanitized HTTP provider-error envelopes.
  - `tests/integration/test_query_api.py`, `tests/unit/test_llm_provider.py`, `tests/unit/test_provider_dependencies.py` — covers mocked query behavior, error responses, adapter translation, and dependency composition.
- Validation:
  - `./.venv/bin/python -m pytest -q` — passed: 22 tests passed
  - `./.venv/bin/ruff check src tests` — passed
  - `./.venv/bin/mypy src` — passed: no issues in 26 source files
  - `git diff --check 2987fe0..HEAD` — passed
  - `git status --short` — passed: clean worktree before this ledger update
- Decisions:
  - LiteLLM remains behind the `ChatProvider` contract; expected provider failures map to fixed, non-sensitive HTTP responses.
- Remaining:
  - Stage 3 ingestion and RAG work has not started. Stage 1 still needs direct runtime dependency pins before it can be marked complete.
- Commit: `3eb1bce`

### 2026-08-03 — Structural digital-PDF normalization

- Stage: Stage 3
- Status: completed
- Scope: Implemented and verified a provider-neutral Docling adapter that converts supported digital PDFs into deterministic structured document elements with hierarchy, page provenance, character offsets, tables, and stable rejection reasons.
- Changed:

  - `requirements.txt` — adds pinned Docling and pypdf runtime dependencies.
  - `src/insightflow/core/exceptions.py` — defines stable document-rejection reasons and the ingestion rejection exception.
  - `src/insightflow/rag/parsers/docling_pdf.py`, `src/insightflow/rag/parsers/__init__.py` — implements and exports PDF inspection, OCR-disabled Docling conversion, structural mapping, deterministic identities, offsets, and normalized output assembly.
  - `src/insightflow/rag/__init__.py` — exposes the PDF parser through the RAG package boundary.
  - `tests/unit/test_docling_pdf_parser.py` — verifies structural mapping, hierarchy, tables, provenance, deterministic output, conversion policy, encryption detection, and rejection behavior with mocked conversion except for converter configuration and pypdf inspection.
- Decisions:

  - The parser supports digital PDFs only: OCR is disabled, table-structure extraction is enabled, and scanned or textless PDFs receive explicit rejection reasons.
  - Docling-specific objects remain behind an adapter and do not cross the provider-neutral RAG domain boundary.
- Remaining:

  - Complete the remaining Stage 3 ingestion path: traceable chunking, hosted embedding execution, Qdrant collection and payload persistence, dense retrieval, grounded generation, structured answers, and persistence verification.
  - Real end-to-end PDF conversion was not validated; conversion behavior is covered primarily with fakes, while real checks cover converter configuration and pypdf inspection.
- Commit: `uncommitted`

### 2026-08-14 — Bounded Max-Min semantic chunking

- Stage: Stage 3
- Status: completed
- Scope: Implemented the reusable semantic chunker component; ingestion orchestration and vector persistence remain deferred.
- Changed:
  - `src/insightflow/rag/chunkers/` — adds English-first sentence segmentation, hosted embedding batches, the published Max-Min clustering rule, bounded split/merge behavior, and deterministic traceable chunks.
  - RAG and application configuration — exposes the strategy and validated algorithm defaults while rejecting unsupported overlap.
  - Embedding and chunker protocols — adds model-aware token counting and makes chunking asynchronous.
  - `notebooks/inspect_max_min_chunking.ipynb` — parses supported documents, makes real hosted sentence-embedding calls, and renders chunks without writing to Qdrant.
  - Unit tests and setup documentation — cover thresholds, batching, connection options, token bounds, structured elements, traceability, deterministic output, and malformed vectors without provider calls.
- Validation:
  - `./.venv/bin/python -m pytest -q` — passed: 78 tests passed
  - `./.venv/bin/ruff check src tests` — passed
  - `./.venv/bin/mypy src` — passed: no issues in 37 source files
  - `git diff --check` — passed before this ledger update
- Decisions:
  - The maximum token limit is strict; the minimum is best effort when no legal adjacent merge fits.
  - Version one uses deterministic English-first sentence rules and the configured hosted embedding model.
- Remaining:
  - Add parser-to-chunker ingestion orchestration, chunk-level embedding and Qdrant persistence, dense retrieval, grounded generation, structured answers, and persistence verification.
  - Validate chunk quality and tune Max-Min thresholds on a representative corpus with real hosted embeddings.
- Commit: `uncommitted`

### 2026-10-26 — Parser-to-chunker ingestion orchestration

- Stage: Stage 3
- Status: partial
- Scope: Connected corpus discovery, MIME-based parser selection, and the configured chunker into one deterministic pipeline. Embedding of chunks and Qdrant persistence remain deferred.
- Changed:
  - `src/insightflow/rag/ingestion.py` — adds `discover_sources`, `create_default_parsers`, `create_chunker`, `IngestionPipeline`, and `IngestedDocument`.
  - `src/insightflow/rag/models.py` — `IngestionResult` now requires `source_identifier` so rejected sources stay traceable.
  - `src/insightflow/rag/chunkers/structure_recursive.py` — `chunk` is now asynchronous to satisfy the `Chunker` protocol.
  - `requirements.txt` — repairs the malformed `tiktoken` line.
  - `tests/unit/test_ingestion_pipeline.py`, `tests/unit/test_structure_recursive_chunker.py`, `tests/unit/test_rag_contracts.py` — cover discovery, rejection mapping, determinism, chunk traceability checks, and strategy selection.
- Validation:
  - `ruff check src tests` — passed
  - `mypy src` — passed: no issues in 38 source files
  - `pytest -q -p no:cacheprovider` — passed: 95 tests passed
- Decisions:
  - Sources are processed sequentially in sorted corpus-relative order. Hidden files are skipped; unsupported files produce `rejected` results with code `unsupported_format`.
  - Only `DocumentRejectedError` becomes a per-document result. Provider and unexpected errors stop the run.
  - A document that yields no chunks is rejected as `empty_document`.
- Remaining:
  - Chunk embedding, Qdrant collection creation and persistence, `skipped` handling for unchanged versions, an ingestion CLI, retrieval, and grounded generation.
  - `create_default_parsers` is not covered by tests, and no real-corpus run was made.
- Commit: `1bb0078` (PR #7)

### 2026-10-26 — Chunk embedding, Qdrant persistence, version skipping, and ingestion CLI

- Stage: Stage 3
- Status: partial
- Scope: Extended `IngestionPipeline` to embed chunks and persist them in Qdrant, skip unchanged document versions, and run from a command-line entry point.
- Changed:
  - `src/insightflow/rag/ingestion.py` — the pipeline now checks for a stored version, chunks, embeds in `embedding_batch_size` batches, validates vector count and size, and calls `ChunkStore.replace_document`; adds `create_index_fingerprint`. `IngestedDocument` was removed; the pipeline returns `IngestionResult`s.
  - `src/insightflow/rag/protocols.py` — adds the `ChunkStore` protocol.
  - `src/insightflow/storage/qdrant.py` — creates or verifies the named-vector collection and keyword payload indexes, maps chunks to payloads and back, detects complete stored versions, and replaces a document's points in `upload_batch_size` batches. Qdrant failures map to `StorageError`.
  - `src/insightflow/providers/embeddings.py` — maps LiteLLM embedding failures to the same sanitized provider errors as chat.
  - `src/insightflow/ingest.py` — new CLI: `python -m insightflow.ingest [CORPUS_DIR] [--force]`.
  - `README.md`, `.env.example`, `CLAUDE.md` — document ingestion usage and required embedding settings.
  - `tests/unit/test_ingestion_pipeline.py`, `tests/unit/test_qdrant_store.py`, `tests/unit/test_ingest_cli.py`, `tests/unit/test_embedding_provider_contract.py` — cover storage, skipping, forced re-ingestion, version replacement, vector-size validation, payload round trips, collection compatibility, CLI output, and error mapping.
- Validation:
  - `ruff check src tests` — passed
  - `mypy src` — passed: no issues in 39 source files
  - `pytest -q -p no:cacheprovider` — passed: 114 tests passed
- Decisions:
  - Unit tests use Qdrant's in-memory local mode and fake embeddings; no server or credentials are needed. Payload indexes have no effect in local mode, so index creation is not verified against a server.
  - `replace_document` deletes the document's other points before upserting. An interrupted write leaves an incomplete version, which the completeness check (stored count equals `document_chunk_count`) reports as missing, so the next run rewrites it.
  - The skip fingerprint covers parser name/version, chunker name/version, all chunking settings, and the embedding fingerprint.
  - An existing collection with a different vector size or distance is refused; there is no automatic migration.
- Remaining:
  - No real run against a Qdrant server or hosted embedding model was performed; "searchable after restart" is unverified.
  - `IngestionConfig.max_retries` is not applied; provider and storage calls are not retried.
  - Documents removed from the corpus are not deleted from Qdrant.
  - Chunk token counts are not checked against `EMBEDDING_MAX_INPUT_TOKENS`.
  - Dense retrieval, grounded generation, and structured answers with sources.
- Commit: `1bb0078` (PR #7)

### 2026-10-26 — Dense retrieval, grounded answers, and persistence verification

- Stage: Stage 3
- Status: completed with mocked models
- Scope: Added the `query embedding → retrieval → grounded generation → structured answer` LangGraph path and verified persistence on a local Qdrant server without the real corpus or hosted-model requests.
- Changed:
  - `src/insightflow/rag/retrieval.py` — adds `DenseRetriever`, `BudgetedContextAssembler`, and `render_context_block`.
  - `src/insightflow/rag/protocols.py`, `src/insightflow/storage/qdrant.py` — adds the `ChunkSearcher` protocol; `QdrantStore.search_chunks` returns domain `RetrievedChunk`s with document-ID and MIME-type filters and an optional score threshold. The old `search` method, which returned Qdrant `ScoredPoint`s, was removed. `mime_type` is now a keyword payload index.
  - `src/insightflow/rag/models.py` — `RetrievedChunk` now requires `source_identifier` and `source_name`.
  - `src/insightflow/agents/` — `AgentState` holds `query`, `context`, and `answer`; adds the `retrieve` node; the `answer` node sends a grounded prompt with numbered passages and returns a fixed answer without a model call when nothing is retrieved; `build_graph` wires `retrieve → answer`.
  - `src/insightflow/api/` — `QueryResponse.sources` is now a list of `SourceReference` objects (breaking change from `list[str]`); new embedding, chunk-searcher, retriever, and context-assembler dependencies; `StorageError` maps to HTTP 503 `storage_unavailable`; provider error messages now say "model provider" because embedding failures share them.
  - `pyproject.toml` — registers the `qdrant` marker.
  - Tests: `tests/unit/test_retrieval.py`, `tests/unit/test_agent_graph.py`, `tests/integration/test_qdrant_persistence.py` (opt-in), and a rewritten `tests/integration/test_query_api.py`.
- Validation:
  - `docker compose up -d qdrant` — Qdrant v1.18.2 started and reported healthy.
  - `ruff check src tests` — passed
  - `mypy src` — passed: no issues in 40 source files
  - `pytest` — passed: 126 passed, 1 skipped (the `qdrant` test without `QDRANT_TEST_URL`)
  - `QDRANT_TEST_URL=http://localhost:6333 pytest -m qdrant` — passed: 1 passed; ingestion, search from a new client, `skipped` on re-ingestion, and payload indexes on the server.
  - Manual restart check (script outside the repository, synthetic documents, fake keyword embeddings): ingested two documents into `insightflow_restart_check`, ran `docker compose restart qdrant`, then retrieval returned `river.txt` at rank 1 with its content and re-ingestion reported both documents `skipped`. The collection was deleted afterwards.
- Decisions:
  - Context tokens are counted with the embedding model's tokenizer as an approximation of the chat model's count.
  - `sources` lists every passage given to the model, numbered to match `[n]` citations; answers are not parsed to filter cited passages.
  - Language and date filters in `RetrievalQuery` raise `ValueError` because stored payloads do not carry those fields.
- Remaining:
  - No ingestion of the full `corpus/` (PDF, DOCX, TXT). See the 2026-10-29 entry for the hosted-model run on `corpus/MD`.
  - Ingestion retries (`max_retries`), deletion of documents removed from the corpus, and chunk-size checks against `EMBEDDING_MAX_INPUT_TOKENS` are not implemented.
  - Stage 1 still lacks pinned versions for several runtime dependencies.
- Commit: `1bb0078` (PR #7)

### 2026-10-29 — Hosted-model check with NVIDIA NIM

- Stage: Stage 3
- Status: completed
- Scope: Verified the chat and embedding settings against NVIDIA NIM, then the author ran ingestion and a query end to end.
- Changed:
  - No code changes. Configuration only, in the untracked `.env`.
- Validation:
  - A diagnostic script outside the repository sent one chat request and two embedding requests with the `.env` settings. Its output was reviewed in the session:
    - `LITELLM_CHAT_MODEL=nvidia/nemotron-3.5-bolt-e4b` — failed: `BadRequestError` "LLM Provider NOT provided"; the API returned 503 `provider_configuration_error`.
    - `LITELLM_CHAT_MODEL=nvidia_nim/nemotron-3.5-bolt-e4b` — failed: `NotFoundError` 404.
    - `LITELLM_CHAT_MODEL=nvidia_nim/nvidia/nemotron-3.5-bolt-e4b` — passed.
    - `LITELLM_EMBEDDING_MODEL=nvidia_nim/nvidia/nemotron-3-embed-1b` — passed with and without `input_type=query`; vector size 2048.
  - Ingestion of `corpus/MD` into the `insightflow_trial` collection and `POST /query` — reported working by the author; the command output was not recorded here.
- Decisions:
  - NVIDIA NIM models need the `nvidia_nim/` LiteLLM prefix followed by the full model ID, for example `nvidia_nim/nvidia/<model>`.
  - `EMBEDDING_DIMENSIONS=2048` for `nemotron-3-embed-1b`.
- Remaining:
  - The chat-provider startup check accepts `nvidia/<model>` because LiteLLM infers the provider from the base URL, but the request then fails. A stricter check that requires an explicit provider prefix is not implemented.
  - Chat and embeddings share `LITELLM_API_KEY` and `LITELLM_API_BASE`; a separate embedding provider needs new settings.
  - Embedding requests do not send `input_type` (`passage` for documents, `query` for questions), which NVIDIA retrieval models can use.
- Commit: `uncommitted`

### 2026-11-01 — Stage 4 agent workflow, conversation threads, and streaming

- Stage: Stage 4
- Status: partial (implementation complete; verified with mocks only)
- Scope: Replaced the fixed `retrieve → answer` graph with a routed, verified workflow with bounded retries; added structured execution metadata, a persistent checkpointer with conversation threads, and an SSE endpoint.
- Changed:
  - `src/insightflow/agents/graph.py` — `route → (retrieve → answer → verify ⇄ rewrite | answer) → finalize` with pure routing functions; `AGENT_MAX_ATTEMPTS` bounds the retry loop; each node is traced in `node_trace`.
  - `src/insightflow/agents/nodes/` — new `route` (LLM JSON: `retrieve`/`direct` and a standalone query), `verify` (citation checks, then an optional LLM judge), `rewrite` (new retrieval query), `finalize` (appends the turn to `history`); `answer` adds history, a direct mode, and token streaming; `retrieve` uses the standalone or rewritten query.
  - `src/insightflow/agents/structured.py`, `models.py`, `state.py`, `checkpoint.py` — prompted JSON parsing with safe fallbacks; workflow models; per-turn state reset (`new_turn_input`); checkpointer factory (`none`/`memory`/`sqlite`) with a serializer allowlist for the state's Pydantic types.
  - `src/insightflow/api/` — `QueryResponse` adds `thread_id`, `verification`, and `metadata` (`answer` and `sources` keep their shape); `QueryRequest` adds `thread_id`; new `POST /query/stream` (FastAPI native SSE), `GET /threads/{id}`, `DELETE /threads/{id}`; `describe_error` shares the sanitized error mapping with the stream; new `thread_not_found` (404).
  - `src/insightflow/main.py` — lifespan opens the checkpointer once and closes it at shutdown.
  - `src/insightflow/providers/llm.py` — `stream` now maps LiteLLM failures (at start and during iteration) to the same sanitized provider errors as `complete`; usage-only stream chunks are skipped.
  - `requirements.txt` — adds `langgraph-checkpoint-sqlite==3.1.3` (brings `aiosqlite`); installing it upgraded `langgraph-checkpoint` from 4.1.1 to 4.3.1.
  - `.env.example`, `.gitignore` (`data/`), `README.md`, `CLAUDE.md` — new settings, endpoints, and workflow description.
  - Tests: new `tests/fakes.py`, `tests/unit/test_agent_nodes.py`, `tests/unit/test_structured_json.py`, `tests/unit/test_app_lifespan.py`, `tests/integration/test_query_stream_api.py`; rewritten `tests/unit/test_agent_graph.py`; extended `tests/integration/test_query_api.py` and `tests/unit/test_llm_provider.py`.
- Validation:
  - `uv pip install --python .venv/bin/python langgraph-checkpoint-sqlite` — installed `langgraph-checkpoint-sqlite` 3.1.3, `aiosqlite` 0.22.1, `sqlite-vec` 0.1.9; upgraded `langgraph-checkpoint` 4.1.1 → 4.3.1.
  - `ruff check src tests` — passed
  - `mypy src` — passed: no issues in 49 source files
  - `pytest -p no:cacheprovider -q` — passed: 215 passed, 1 skipped (the `qdrant` test without `QDRANT_TEST_URL`); coverage 92%, every `agents/` module at 92–100%.
  - `git diff --check` — passed
  - `CHECKPOINT_SQLITE_PATH=<scratch>/checkpoints.sqlite uvicorn insightflow.main:app --app-dir src --port 8765` — the lifespan created the SQLite file; `GET /openapi.json` returned 200, `GET /threads/unknown` returned 404 `thread_not_found`, and `POST /query` with `thread_id` "bad id" returned 422. No query was sent to a model.
- Decisions:
  - LLM nodes ask for JSON in the prompt instead of `response_format`, so every LiteLLM provider works. Unreadable router output falls back to retrieval; unreadable judge output gives `unchecked`; unreadable rewrite output reuses the question.
  - An answer with no citation fails verification and triggers a retry, including "I do not know" answers, because a new retrieval query can find the missing context. Empty retrieval is `skipped` and is not retried.
  - When retries are used up, the last answer is returned with `verification.status = "failed"`; it is not hidden.
  - SQLite is the default checkpointer, so the runtime stays the application plus Qdrant. LangGraph's msgpack serializer restores only allow-listed Pydantic types (`CHECKPOINT_STATE_TYPES`); without the allowlist it logs a deprecation warning, and its strict mode would return dicts.
  - Concurrent requests on the same thread are not serialized.
- Remaining:
  - No run with a hosted model: router and judge prompt quality, JSON reliability on NVIDIA NIM, streaming with LiteLLM, and the real cost and latency of the extra calls are unverified.
  - Threads have no owner and no expiry; authentication is Stage 8 work.
  - Parallel retrieval for decomposed queries and human approval (optional Stage 4 work) are not implemented.
- Commit: `eef4a6f`

### 2026-11-04 — Stage 5 MCP client boundary and calculator server

- Stage: Stage 5
- Status: partial (steps 1–3 of 6: branch, MCP client boundary, calculator server)
- Scope: Added the official MCP SDK, a provider-neutral `ToolClient` contract with an MCP adapter that applies the per-call controls from the plan, and an in-repo calculator MCP server. The workflow does not call tools yet.
- Changed:
  - `src/insightflow/tools/models.py`, `protocols.py` — `ToolSpec`, `ToolResult` (call ID, server, tool, arguments, status, stable error code, output, structured output, `truncated`, start time, duration), and the `ToolClient` protocol.
  - `src/insightflow/tools/mcp_client.py` — `McpToolClient` for an in-process server, a stdio subprocess, a transport, or a URL. Allowlist; argument check against the tool's input schema before the call (`jsonschema`); one timeout for each list or call operation; text output limit and dropping of oversized structured output; failures returned as `ToolResult` codes (`unknown_tool`, `invalid_arguments`, `tool_failed`, `timeout`, `unavailable`) without server or transport exception text. The tool list is cached per client; each operation opens its own connection. Logs record only server, tool, status, error code, and duration.
  - `src/insightflow/tools/calculator.py`, `calculator_server.py` — `ast`-based evaluator (no `eval`) with allow-listed operators, functions, and constants, an expression limit of 200 characters, an integer exponent limit, and a result limit of 10^100; MCP server `insightflow-calculator` with the read-only tool `calculate`, runnable over stdio.
  - `src/insightflow/core/exceptions.py` — `ToolError`, `ToolTimeoutError` (raised only by `list_tools`).
  - `requirements.txt` — adds `mcp==2.3.0` (released 2026-10-02) and `jsonschema` (imported directly; already a dependency of `mcp`). `requirements-dev.txt` — adds `types-jsonschema`.
  - `README.md`, `CLAUDE.md`, `docs/Implement Plan Draft.md` — tools layer, calculator usage, and the selected Stage 5 tools.
  - Tests: new `tests/unit/test_calculator.py`, `tests/unit/test_mcp_tool_client.py` (in-process `MCPServer` instances and failing or hanging fake transports), `tests/integration/test_calculator_stdio.py` (real subprocess over stdio).
- Validation:
  - `uv pip install --python .venv/bin/python "mcp==2.3.0"` — installed `mcp` 2.3.0, `mcp-types` 2.3.0, `httpx2` 2.16.0, `httpcore2` 2.16.0, `pyjwt` 2.15.1, `cryptography` 50.0.2, `cffi` 2.1.1, `pycparser` 3.12, `opentelemetry-api` 1.45.2, `python-multipart` 0.0.33, `sse-starlette` 3.6.0, `truststore` 0.10.5.
  - `uv pip install --python .venv/bin/python types-jsonschema` — installed `types-jsonschema` 4.26.0.20261006.
  - `ruff check src tests` — passed
  - `mypy src` — passed: no issues in 55 source files
  - `pytest -p no:cacheprovider -q` — passed: 292 passed, 1 skipped (the `qdrant` test without `QDRANT_TEST_URL`); coverage 93%; `tools/` modules at 90–100% (`mcp_client.py` 98%, `calculator.py` 97%).
  - `git diff --check` — passed
- Decisions:
  - InsightFlow is an MCP client. Exposing InsightFlow as an MCP server is out of scope.
  - `call_tool` returns failures as results so that the workflow can record the attempt and let the model explain it; `list_tools` raises, so that a node can skip the tool route when a server is unusable.
  - Arguments are validated on the client side: the MCP 2.x client validates tool output but not input, and the server's own validation error contains raw pydantic text.
  - The MCP client response cache is disabled; the adapter caches the tool list itself, and tool results are never cached.
  - A server error from a tool (`is_error`) is passed to the model as a length-limited message (for example "division by zero"); unexpected server exceptions reach the client only as a generic message from the SDK.
  - Prompt-injection controls for web results are Stage 8 work, by the author's decision.
- Remaining:
  - Step 4: settings (enable flag, timeout, output limit), dependency wiring, a `tool` route in the router, a tool node, tool results in the answer prompt and execution metadata, and routing and termination tests.
  - Step 4 must configure application logging before an in-process `MCPServer` is created: `MCPServer()` calls `logging.basicConfig`, and the app does not configure the root logger now, so the SDK would install its Rich handler.
  - Step 5: Tavily web search through its hosted MCP endpoint, a web source type with URLs, and an opt-in real-call test.
  - No hosted-model run for Stage 4 or Stage 5.
- Commit: `2a57796`

### 2026-11-04 — Stage 5 ReAct agent workflow and Tavily web search

- Stage: Stage 5
- Status: partial (steps 4–5 implemented; verified with mocks and in-process MCP servers only)
- Scope: Replaced the Stage 4 routed workflow with a ReAct loop in which the agent can use any tool at any step, document search included; added a tool set over several MCP servers, Tavily web search, one source numbering for documents and web pages, and per-step audit metadata. A first version with a router `tool` branch and a separate tool node was built and then replaced by the author's decision before it was committed: it allowed tools only on one branch.
- Changed:
  - `src/insightflow/agents/graph.py` — `agent ⇄ act → answer → verify ⇄ retry → finalize`; direct replies (no observations) skip verification. `recursion_limit(max_steps, max_attempts)` gives the LangGraph step limit for the bounded loops.
  - `src/insightflow/agents/nodes/agent.py` — new: one prompted-JSON `AgentAction` per step from the question, history, catalog with input schemas, observations, verification feedback, and the steps left. Without a model call it answers when `AGENT_MAX_STEPS` actions were made in the attempt or when the action repeats an earlier call (same tool and arguments, ignoring case and spaces). Unreadable output searches the question once, then answers.
  - `src/insightflow/agents/nodes/act.py`, `retry.py` — new: `act` runs the pending action (local search or the tool set) and records an `Observation`; `retry` starts the next attempt and sends the verification issues to the agent.
  - `src/insightflow/agents/document_search.py` — new: `search_documents` as a local tool over `Retriever` and `ContextAssembler`, with a JSON input schema; the name is reserved, and MCP tools with it are hidden. Storage and embedding errors still raise.
  - `src/insightflow/agents/observations.py` — new: `number_sources` numbers passages and web pages once per turn in order of first observation; `render_observations`, `render_tool_catalog`, and `has_evidence` (sources, tool outputs, and errors reported by a tool count; timeouts, unavailable servers, unknown tools, invalid arguments, and empty searches do not).
  - `src/insightflow/agents/nodes/answer.py`, `verify.py` — answer and judge use all observations; citations are checked against all numbered sources; answers without sources are `skipped`.
  - Removed: `agents/nodes/route.py`, `rewrite.py`, `retrieve.py`, `RouteDecision`, `RewrittenQuery`, and the setting `AGENT_ROUTER_ENABLED`.
  - `src/insightflow/agents/state.py`, `models.py`, `checkpoint.py` — per-turn `observations`, `pending_action`, and `feedback`; `CHECKPOINT_STATE_TYPES` is now `AgentAction`, `ConversationTurn`, `Observation`, `VerificationResult`.
  - `src/insightflow/tools/registry.py`, `web_search.py`, `validation.py` — `ToolSet` (one catalog over several servers, failing servers left out, first server keeps a duplicate name, fixed arguments, web-source parser); Tavily registration (`tavily_search` only, `max_results` fixed to `WEB_SEARCH_MAX_RESULTS`, raw content and images off, key in an `Authorization: Bearer` header) and `parse_tavily_sources`; `argument_problem` moved out of `mcp_client.py`.
  - `src/insightflow/tools/mcp_client.py`, `models.py` — transport factories, `http_transport(url, headers)` over `httpx2`, `WebSource`, and `ToolResult.web_sources`.
  - `src/insightflow/api/` — `sources` discriminated by `kind` (`document`, `web`); `metadata.route` removed (breaking); new `metadata.steps` (`StepReport`: attempt, thought, tool, server, arguments, call ID, status, error, truncated, source numbers, duration); `retrieval_queries` now lists the `search_documents` queries; `resolve_thread` always returns a run config with `recursion_limit`; stream `node` events for `agent`, `act`, and `retry`; `create_tool_set` and `get_tool_set`.
  - `src/insightflow/main.py` — the lifespan calls `configure_logging` and then builds the tool set once.
  - `src/insightflow/core/config.py`, `.env.example` — `AGENT_MAX_STEPS` (default 4, 1–8), `TOOL_CALCULATOR_ENABLED`, `TOOL_WEB_SEARCH_ENABLED`, `TOOL_TIMEOUT_SECONDS`, `TOOL_MAX_OUTPUT_CHARS`, `TAVILY_API_KEY` (`SecretStr`), `TAVILY_MCP_URL`, `WEB_SEARCH_MAX_RESULTS`; web search without a key fails at startup.
  - `requirements.txt` — adds `httpx2` (imported directly; already a dependency of `mcp`). `pyproject.toml` — new `tavily` marker.
  - `README.md`, `CLAUDE.md`, `docs/Implement Plan Draft.md` — ReAct workflow, bounds, cost, response shape, tools, and the pattern decision.
  - Tests: rewritten `tests/unit/test_agent_graph.py` and `tests/unit/test_agent_nodes.py`; new `tests/unit/test_tool_registry.py`, `tests/integration/test_query_tools_api.py`, `tests/integration/test_tavily_search.py` (opt-in); updated `tests/fakes.py` (roles `agent`, `answer`, `direct`, `judge`; `FakeToolClient`, `TAVILY_TEXT`, `search()`, `action()`, `ANSWER`), `tests/unit/test_structured_json.py`, `tests/unit/test_mcp_tool_client.py`, `tests/integration/test_query_api.py`, `tests/integration/test_query_stream_api.py`.
- Validation:
  - `ruff check src tests` — passed
  - `mypy src` — passed: no issues in 60 source files
  - `pytest -p no:cacheprovider -q` — passed: 353 passed, 2 skipped (the `qdrant` and `tavily` opt-in tests); coverage 94%; every `agents/` module at 92–100%, `tools/registry.py`, `tools/web_search.py`, and `tools/validation.py` at 100%. The tests include a worst case (an agent that never stops and answers that always fail) for `(max_steps, max_attempts)` = (1, 1), (2, 3), (4, 2), (8, 4), and the default bounds through the API.
  - `pytest -m tavily -p no:cacheprovider -q --no-cov` — 1 skipped, 354 deselected (no `TAVILY_TEST_API_KEY`); the real Tavily call was not run.
  - `git diff --check` — passed
  - `CHECKPOINT_BACKEND=none TOOL_WEB_SEARCH_ENABLED=false uvicorn insightflow.main:app --app-dir src --port 8767` (stopped with `kill`) — startup and shutdown completed; `GET /health/live` and `GET /openapi.json` returned 200; `ExecutionMetadata` has `attempts`, `duration_ms`, `nodes`, `retrieval_queries`, `steps`. No query was sent to a model.
- Decisions:
  - ReAct with prompted JSON, not native function calling: LiteLLM's registry reports that the configured model (`nvidia_nim/nvidia/nemotron-3.5-bolt-e4b`) does not support function calling. Native calls can be added behind `ChatProvider`.
  - The agent only decides and a separate `answer` node writes the answer, so token streaming and the grounded answer prompt are kept; it costs one call per attempt.
  - Document search is a local tool, not MCP, because its result carries passages with provenance for citations.
  - The agent writes its own standalone search queries from the history, so the router is not needed; a greeting is one agent call that answers.
  - Sources are numbered once per turn across steps and attempts; a repeated passage or page keeps its first number.
  - Worst-case cost per question is `AGENT_MAX_ATTEMPTS × (AGENT_MAX_STEPS + 2)` chat calls (12 with the defaults); Stage 4 had at most 6. Prompt size grows with each search, up to `AGENT_MAX_STEPS` × the per-search context budget.
  - A tool server whose listing fails is tried again on the next agent step, and each try can take up to `TOOL_TIMEOUT_SECONDS`.
  - Web content goes to the model without prompt-injection filtering; by the author's decision, that is Stage 8 work.
- Remaining:
  - No hosted-model run: the agent's JSON reliability on NVIDIA NIM, whether it searches before it answers factual questions, its tool choices, and the real cost and latency are unverified. Run `PYTHONPATH=src python scripts/manual_chat_test.py` and real queries before Stage 5 is marked complete.
  - No real Tavily call: the bearer header format (not given by the Tavily docs), the `tavily_search` input schema (taken from the `tavily-mcp` source), and the text format read by `parse_tavily_sources` are unverified. Run `TAVILY_TEST_API_KEY=... pytest -m tavily`.
  - `metadata.route` was removed; API clients that read it must change.
  - Caching of failed tool listings and a total context budget across searches are not implemented.

### 2026-11-10 — Stage 5 real Tavily search and hosted-model runs

- Stage: Stage 5
- Status: partial (web search verified against the real Tavily server and the hosted model; not committed)
- Scope: First real runs of the ReAct workflow with web search on. They found and fixed a bug that lost all web sources of a large search result.
- Bug: Tavily's hosted server returns one JSON object (`results` with `url`, `title`, `content`, `score`, `raw_content`, `id`) as structured content and as text. With 5 results, the object was about 7,300 characters, over `TOOL_MAX_OUTPUT_CHARS=6000`. `McpToolClient` cut the text and dropped the structured data at the same limit, so `parse_tavily_sources` found no pages: the response had `sources: []`, and the model cited Tavily's `id` values (for example `[50469a-01]`) instead of `[n]`.
- Changed:
  - `src/insightflow/tools/mcp_client.py` — structured output has its own limit, `max_structured_chars` (default `MAX_STRUCTURED_CHARS = 100_000`), because code reads it and the model does not; text keeps `max_output_chars`. Structured data over its limit is dropped, not cut.
  - `src/insightflow/tools/web_search.py` — `parse_tavily_sources` also reads the JSON text when structured content is missing, and takes `max_page_chars`; the Tavily registration sets it to `max(200, TOOL_MAX_OUTPUT_CHARS // WEB_SEARCH_MAX_RESULTS)`, so the pages of one search stay near the output limit.
  - Tests: `tests/unit/test_mcp_tool_client.py` (structured data kept over the text limit, dropped over its own limit, invalid `max_structured_chars`); `tests/unit/test_tool_registry.py` (JSON text, cut JSON, page clipping, the registration's page limit, and a regression test with an in-process server that returns five 1,500-character pages).
- Validation:
  - `TAVILY_TEST_API_KEY=<from .env> pytest -m tavily -p no:cacheprovider --no-cov -v` — passed (1 passed, 354 deselected). The bearer header is accepted.
  - `ruff check src tests` — passed; `mypy src` — passed: no issues in 60 source files
  - `pytest -q -p no:cacheprovider` — passed: 359 passed, 2 skipped; coverage 94%
  - `uvicorn insightflow.main:app --app-dir src --port 8000` with `TOOL_WEB_SEARCH_ENABLED=true`, Qdrant collection `insightflow_trial` (28 points), chat model `nvidia_nim/nvidia/nemotron-3.5-bolt-e4b`:
    - Before the fix, `POST /query` "What is the latest FastAPI release and when was it published?": two `tavily_search` steps, both `truncated`, `sources: []`, verification `skipped`, 12.5 s.
    - After the fix, the same question: one `tavily_search` step with sources [1]–[5] (all `web`, with URLs), answer cites [3] and [5], verification `passed` (`llm`), 6.8 s.
    - `POST /query/stream`, a question on the docs, the newest version, and a version count: `search_documents` (sources [1]–[6], `document`) then `tavily_search` ([7]–[11], `web`), one numbering; verification `passed`; 11.6 s; stream `node` events for `agent` and `act` arrived before the tokens.
    - Server log: one `tool call server=tavily tool=tavily_search status=ok` line per search, with no arguments or key; no warnings.
- Observed model behavior (not fixed):
  - In the mixed question, the agent did not call `calculate` for the version count; the answer said that it did not know.
  - The answers gave 0.162.0 as the newest version, but one search result in the same session showed 0.162.2 on the GitHub releases page. The judge checks that claims are supported by the sources, not that the sources are current.
- Remaining: commit; decide whether the agent prompt needs a stronger rule to calculate with numbers from observations.

### 2026-11-10 — Stage 5 tool server cooldown

- Stage: Stage 5
- Status: partial (implemented and tested; not committed)
- Scope: A tool server that is down no longer costs a timeout on each agent step. Before, `ToolSet.list_tools` tried a failed server again on every step, and each try could take up to `TOOL_TIMEOUT_SECONDS` (20 s by default, so up to 80 s for 4 steps).
- Changed:
  - `src/insightflow/tools/registry.py` — `ToolSet(failure_cooldown_seconds=60, clock=time.monotonic)`. A failed listing, or a call that fails with `timeout` or `unavailable` (`PAUSING_ERRORS`), pauses the server. While paused, the catalog leaves it out and its calls return `unavailable` with `PAUSED_MESSAGE` and no connection. The first listing after the cooldown tries the server again; a successful listing ends the pause. `tool_failed` and `invalid_arguments` do not pause. 0 turns pauses off. Log line: `tool server paused server=... reason=... cooldown_s=...`.
  - `src/insightflow/core/config.py`, `.env.example`, `src/insightflow/api/dependencies.py` — `TOOL_FAILURE_COOLDOWN_SECONDS` (default 60, 0–3600) is passed to the tool set.
  - `README.md`, `CLAUDE.md` — the pause behavior.
  - Tests: `tests/fakes.py` (`FakeToolClient.call_error` and `list_calls`); `tests/unit/test_tool_registry.py` (listing pause and retry after the cooldown, pause after `timeout` and `unavailable`, no pause after `tool_failed` and `invalid_arguments`, cooldown 0, negative cooldown, the setting); `tests/integration/test_query_tools_api.py` (two questions with three agent steps each list a failing server once).
- Validation:
  - `ruff check src tests` — passed; `mypy src` — passed: no issues in 60 source files
  - `pytest -q -p no:cacheprovider` — passed: 368 passed, 2 skipped; coverage 94%; `tools/registry.py` 100%
  - `PYTHONPATH=src python <scratch>/cooldown_check.py` — the real Tavily registration against `http://10.255.255.1/mcp/` (no answer) with a 2 s timeout and four listings: step 1 took 2.01 s and logged the pause; steps 2–4 took 0.00 s. No model or Tavily call.
- Decisions:
  - The pause is in `ToolSet`, not in `McpToolClient`: the set sees both failed listings and failed calls after a cached listing, and the app keeps one set for all requests.
  - One `timeout` pauses the server for all requests. A slow but working server can be hidden for one cooldown; set `TOOL_FAILURE_COOLDOWN_SECONDS=0` to turn this off.
- Remaining: commit; the calculator rule in the agent prompt; a total context budget across searches.

### 2026-11-10 — Stage 5 closed

- Stage: Stage 5
- Status: completed
- Scope: Closed Stage 5 by the author's decision. Every core item of the plan has evidence: the MCP SDK (`mcp==2.3.0`), the client boundary (`ToolClient`, `McpToolClient`), one low-risk tool (calculator) before the registry (`ToolSet`), input schemas, timeouts, error codes, output limits, audit metadata (`metadata.steps`, outcome-only log lines), and tests that the graph always stops. The router was replaced by a ReAct loop, so "tool/no-tool routing" is covered by the agent tests for tool, no-tool, and direct answers.
- Commits: `2a57796` (MCP client boundary and calculator server), `79a30f6` (ReAct workflow and Tavily web search, with the fix for lost web sources), and `6852a03` (tool server cooldown and this entry).
- Validation (2026-11-10, before the closing commit):
  - `ruff check src tests` — passed
  - `mypy src` — passed: no issues in 60 source files
  - `pytest -q -p no:cacheprovider` — passed: 368 passed, 2 skipped; coverage 94%
  - `pytest -m tavily -p no:cacheprovider --no-cov -q` — 1 skipped without `TAVILY_TEST_API_KEY`; the same test passed earlier on 2026-11-10 with the key (see "Stage 5 real Tavily search and hosted-model runs")
  - `git diff --check` — passed
- Known limits carried forward:
  - In a mixed question, the hosted model did not call `calculate` for a count; a stronger calculator rule in the agent prompt is not implemented.
  - No total context budget across searches: each search adds its passages to the prompt, up to `AGENT_MAX_STEPS` searches per attempt.
  - `metadata.route` was removed (breaking change for API clients).
  - Web content goes to the model without prompt-injection filtering (Stage 8, by the author's decision).
  - The verification retry path was not run against the hosted model; it is covered by mocked tests.
  - Optional plan work is not implemented: dynamic tool registration, auth propagation, result caching, and more MCP servers.

### 2026-11-13 — Stage 6 retrieval-quality options and evaluation set

- Stage: Stage 6
- Status: in progress (code complete; measured in part, see the next entry)
- Scope: Implemented every Stage 6 candidate as an option that is off by default, a labeled evaluation set for the sample corpus, and a command that compares retrieval configurations. Default behavior is unchanged: plain dense retrieval, the same agent prompt, and the same index fingerprint for dense-only indexes.
- Changed:
  - `src/insightflow/rag/sparse.py` — new: `Bm25SparseEncoder` (hashed terms, BM25 term weights, no model or vocabulary).
  - `src/insightflow/rag/retrieval.py` — `HybridRetriever`, `renumber`; `DenseRetriever` passes `source_identifiers`.
  - `src/insightflow/rag/query_expansion.py` — new: `QueryExpandingRetriever` (`multi_query`, `decompose`), `reciprocal_rank_fusion`.
  - `src/insightflow/rag/corrective.py` — new: `CorrectiveRetriever` (grade, drop, one more search).
  - `src/insightflow/rag/refinement.py` — new: `RefiningRetriever`, `Reranker`, `NeighborExpander`, `ContextCompressor`.
  - `src/insightflow/rag/composition.py` — new: `build_retriever` from `RetrievalConfig`.
  - `src/insightflow/rag/config.py`, `models.py`, `protocols.py` — `RetrievalConfig` options (the unused `expand_neighbors`, `max_neighbors_per_hit`, and `per_document_cap` were replaced), `SparseVector`, `RetrievalQuery.source_identifiers`, `CandidateProcessor`, `SparseEncoder`, `ChunkReader`; `ChunkStore` and `ChunkSearcher` accept sparse vectors.
  - `src/insightflow/rag/ingestion.py`, `src/insightflow/ingest.py` — sparse vectors and a fingerprint part for the encoder, only in hybrid mode.
  - `src/insightflow/storage/qdrant.py` — sparse vector with the IDF modifier, hybrid search (prefetch and RRF), source filter, `get_chunks`.
  - `src/insightflow/storage/qdrant_memory.py`, `src/insightflow/agents/memory.py`, `src/insightflow/agents/nodes/recall.py`, `finalize.py`, `graph.py`, `state.py` — conversation memory, the optional `recall` node, `thread_id`, `search_scope`, and `recalled` in the state.
  - `src/insightflow/providers/rerank.py` — new: `RerankProvider`, `LiteLLMRerankProvider`. `providers/structured.py` — moved from `agents/structured.py` (a re-export stays).
  - `src/insightflow/agents/document_search.py`, `nodes/agent.py`, `nodes/act.py` — filter arguments, request scope, multi-hop hint.
  - `src/insightflow/api/` — `QueryRequest.filters`, `metadata.recalled_turns`, `get_retriever` composition, `get_rerank_provider`, `get_conversation_memory`; `DELETE /threads/{id}` also deletes remembered turns.
  - `src/insightflow/evaluation/retrieval.py`, `src/insightflow/evaluate.py`, `evaluation/retrieval_cases.jsonl` — new: scoring, the comparison command, and 48 labeled questions.
  - `src/insightflow/core/config.py`, `.env.example`, `README.md`, `CLAUDE.md`, `docs/Implement Plan Draft.md` — settings and documentation.
  - Tests: 11 new unit test files, additions to `tests/fakes.py`, `tests/integration/test_query_api.py`, and `tests/integration/test_qdrant_persistence.py`.
- Validation:
  - `ruff check src tests` — passed
  - `mypy src` — passed: no issues in 73 source files
  - `pytest -q -p no:cacheprovider` — passed: 482 passed, 4 skipped; coverage 95%
  - `docker compose up -d qdrant`, then `QDRANT_TEST_URL=http://localhost:6333 pytest -m qdrant -q -p no:cacheprovider --no-cov` — passed: 3 passed (persistence, hybrid search with filters and chunk reads, conversation memory). Synthetic documents and fake embeddings; no hosted-model request.
  - `git diff --check` — passed
  - Evaluation labels: the sample corpus was parsed and chunked locally with the default parsers and `structure_recursive` settings (no hosted request), and every evidence text of the 48 cases was found in at least one chunk of its source (scratch script, not in the repository). The default suite checks that the set loads and that each source file exists.
- Findings:
  - The first server run of the hybrid test failed: with tied dense scores, rank fusion returned a document that was first in one ranking only, and the in-memory test passed by chance. `search_chunks` now reads each ranking to `max(2 × limit, 20)`, and the tests use a corpus in which dense search ranks the target second.
  - Local chunking of the sample PDFs gives many small chunks (for example 42 chunks for 8,076 characters in the Computer Science catalog), because each heading starts a chunk. This is a reason to measure `RAG_NEIGHBORS_PER_HIT` first; it is not measured yet.
- Decisions:
  - Every option is off by default. The plan says to compare each addition on the evaluation set before it enters the default path, and no hosted comparison is recorded.
  - Sparse vectors are computed in Python and Qdrant applies IDF, so hybrid retrieval adds no model download (FastEmbed stays out).
  - Parent context uses the stored neighbor links, so the chunkers and existing indexes do not change. A parent-child hierarchy in the index is not implemented.
  - Compression is extractive (sentence numbers), so the model cannot add text; an unreadable or empty selection keeps the passage.
  - Memory failures are logged and do not fail a request; memory needs a checkpointer because it is per thread.
  - An existing dense-only collection is refused in hybrid mode, not migrated.
- Not verified:
  - No hosted-model or hosted-reranker request was made. The scores of each option on the evaluation set, the JSON reliability of the expansion, grading, and compression prompts on the configured NVIDIA NIM model, the LiteLLM rerank call against a real provider, and real cost and latency are unknown.
  - The agent's use of the filter arguments and of the multi-hop hint is tested with scripted replies only.
- Remaining:
  - Ingest the corpus in hybrid mode into a new collection and run `python -m insightflow.evaluate`; record the table here; then decide the defaults.
  - Carried from Stage 5: a total context budget across the searches of one turn; the calculator rule in the agent prompt.

### 2026-11-13 — Stage 6 measurement with hosted embeddings and the embedding input-type fix

- Stage: Stage 6
- Status: in progress
- Scope: Ran the labeled comparison for the options that need embeddings only (approved by the author; no chat or rerank calls). The first run showed that dense retrieval was almost random. The cause was found and fixed, and the comparison was repeated.
- Setup: local Qdrant (`docker compose up -d qdrant`), a new collection `insightflow_eval`, the sample corpus (50 documents, all `completed`), `structure_recursive` chunking with the default settings, embedding model `nvidia_nim/nvidia/nemotron-3-embed-1b` (2048 dimensions), `RAG_CANDIDATE_TOP_K=12`, `RAG_GENERATION_MAX_CHUNKS=6`, `evaluation/retrieval_cases.jsonl` (48 cases). The scores are for the passages in the assembled context.
- Run 1 (no `input_type`, the behavior before this change):
  - `RAG_RETRIEVAL_MODE=hybrid QDRANT_COLLECTION=insightflow_eval PYTHONPATH=src python -m insightflow.ingest corpus/` — 50 documents: 50 completed
  - `QDRANT_COLLECTION=insightflow_eval PYTHONPATH=src python -m insightflow.evaluate --variants dense,hybrid,dense+neighbors,hybrid+neighbors --output <scratch>/scores.json`

    | Variant | hit | recall | mrr | context tokens |
    | --- | --- | --- | --- | --- |
    | dense | 0.500 | 0.469 | 0.398 | 1818 |
    | hybrid | 0.833 | 0.823 | 0.710 | 1940 |
    | dense+neighbors | 0.500 | 0.490 | 0.415 | 3519 |
    | hybrid+neighbors | 0.833 | 0.823 | 0.689 | 3688 |

  - A probe of two missed questions showed dense scores between 0.80 and 0.83 for unrelated passages (for example a `TXT/dracula.txt` passage for a question about the meal allowance).
- Cause: the embedding model is asymmetric, and `LiteLLMEmbeddingProvider` sent no `input_type`. Check with one question, its answer passage, and one unrelated passage (`<scratch>/input_type.py`, four embedding requests): without `input_type` the cosine similarity was 0.714 (relevant) and 0.746 (unrelated); with `query` and `passage` it was 0.456 and 0.005.
- Changed:
  - `src/insightflow/providers/embeddings.py` — `query_input_type` and `document_input_type`; `embed_query` no longer calls `embed_documents`.
  - `src/insightflow/core/config.py`, `.env.example` — `EMBEDDING_QUERY_INPUT_TYPE`, `EMBEDDING_DOCUMENT_INPUT_TYPE` (empty by default: nothing is sent, as before).
  - `src/insightflow/rag/config.py`, `src/insightflow/ingest.py` — `EmbeddingConfig.document_input_type` is part of the fingerprint only when set, so a change re-embeds the documents and older indexes keep their fingerprint.
  - `src/insightflow/api/dependencies.py` — passes both types.
  - Tests: `tests/unit/test_embedding_provider_contract.py`, `tests/unit/test_ingest_cli.py`.
- Run 2 (`EMBEDDING_QUERY_INPUT_TYPE=query EMBEDDING_DOCUMENT_INPUT_TYPE=passage` for both commands; the ingestion embedded all 50 documents again because the fingerprint changed):

    | Variant | hit | recall | mrr | context tokens |
    | --- | --- | --- | --- | --- |
    | dense | 0.979 | 0.979 | 0.915 | 1147 |
    | hybrid | 0.979 | 0.958 | 0.924 | 1725 |
    | dense+neighbors | 0.979 | 0.979 | 0.915 | 1624 |
    | hybrid+neighbors | 0.979 | 0.958 | 0.913 | 3083 |

  - Dense misses one case (`calendar-registrar-office`), which hybrid finds. Hybrid loses one labeled passage in three cases (`research-cii-projects`, `hop-health-center-hours`, `hop-insurance-and-work`).
- Validation after the fix:
  - `ruff check src tests` — passed; `mypy src` — passed: no issues in 73 source files
  - `pytest -q -p no:cacheprovider --no-cov` — passed: 485 passed, 4 skipped
- Decisions:
  - Dense retrieval stays the default. After the fix, hybrid and neighbor expansion give no gain on this set, and neighbor expansion adds 40% to 80% more context tokens.
  - The input types stay empty by default, because a symmetric model (for example OpenAI) does not accept them. `.env.example` explains when to set them.
- Limits of this measurement:
  - The set is almost solved by dense retrieval (47 of 48), so it can no longer separate the options. Harder cases are necessary before hybrid, neighbors, or reranking can be judged.
  - One corpus, one embedding model, one chunking setup. The labels were written from the corpus by the assistant and were not reviewed by a second person.
  - The scores measure retrieval, not answer quality.
- Not verified: query expansion (`multi_query`, `decompose`), corrective retrieval, context compression, reranking, the multi-hop hint, the agent's filter arguments, and conversation memory on hosted models.
- Remaining:
  - The author's `.env` does not set the input types, so the running application still has the Run 1 behavior. Add `EMBEDDING_QUERY_INPUT_TYPE=query` and `EMBEDDING_DOCUMENT_INPUT_TYPE=passage`, then ingest again.
  - Add harder evaluation cases; measure the chat-backed options and a reranker (billable chat and rerank calls).

### 2026-11-13 — Stage 6 closed

- Stage: Stage 6
- Status: completed
- Scope: Closed Stage 6 by the author's decision. Every candidate of the plan is implemented behind a setting that is off by default, a labeled evaluation set and a comparison command exist, and the one measured failure (missing embedding `input_type`) is fixed. No option was made a default, because the measurement gave no reason for it.
- Validation (2026-11-13, before the closing commit):
  - `ruff check src tests` — passed
  - `mypy src` — passed: no issues in 73 source files
  - `pytest -q -p no:cacheprovider` — passed: 485 passed, 4 skipped; coverage 95%
  - `QDRANT_TEST_URL=http://localhost:6333 pytest -m qdrant -q -p no:cacheprovider --no-cov` — passed: 3 passed
  - `git diff --check` — passed
- Known limits carried forward:
  - Query expansion, corrective retrieval, context compression, reranking, the multi-hop hint, the agent's filter arguments, and conversation memory are verified with mocks only; no hosted chat or rerank request was made for them.
  - The evaluation set is almost solved by dense retrieval (47 of 48), so it cannot separate the options. Harder cases are necessary.
  - The author's `.env` must set `EMBEDDING_QUERY_INPUT_TYPE=query` and `EMBEDDING_DOCUMENT_INPUT_TYPE=passage`, followed by a new ingestion.
  - No parent-child hierarchy in the index, no language or date filters, no total context budget across the searches of one turn.
  - `metadata.recalled_turns` and `QueryRequest.filters` are new API fields.
