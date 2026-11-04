# InsightFlow Progress

## Current status

- Current stage: Stage 5 (MCP client boundary and calculator server on `feature/stage5-mcp-tools`; not wired into the workflow)
- Last updated: 2026-11-04
- Overall state: in progress

## Stage summary

| Stage   | Status      | Evidence                                                                                                    |
| ------- | ----------- | ----------------------------------------------------------------------------------------------------------- |
| Stage 1 | in progress | Repository foundation is present, but tests, runtime dependency pins, and mypy validation remain incomplete |
| Stage 2 | complete    | Query workflow, provider validation and error mapping, and mocked contract coverage verified                |
| Stage 3 | complete | Ingestion, Qdrant persistence, dense retrieval, and grounded answers with sources verified with mocks and on a real Qdrant server; a hosted-model run on `corpus/MD` was reported working by the author (2026-10-29) |
| Stage 4 | in progress | Router, verifier, bounded retries, execution metadata, SQLite checkpointer with threads, and SSE streaming verified with mocks; no hosted-model run yet |
| Stage 5 | in progress | MCP client adapter with allowlist, input-schema, timeout, and output-size controls, and a calculator MCP server, verified in-process and over stdio; no graph integration or web search yet |
| Stage 6 | not started | —                                                                                                          |
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
