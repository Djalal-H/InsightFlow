# InsightFlow Progress

## Current status

- Current stage: Stage 3 (implementation complete; hosted-model run pending)
- Last updated: 2026-10-26
- Overall state: in progress

## Stage summary

| Stage   | Status      | Evidence                                                                                                    |
| ------- | ----------- | ----------------------------------------------------------------------------------------------------------- |
| Stage 1 | in progress | Repository foundation is present, but tests, runtime dependency pins, and mypy validation remain incomplete |
| Stage 2 | complete    | Query workflow, provider validation and error mapping, and mocked contract coverage verified                |
| Stage 3 | complete (mocked models) | Ingestion, Qdrant persistence, dense retrieval, and grounded answers with sources verified; restart persistence verified on a real Qdrant server with fake embeddings; no hosted-model run |
| Stage 4 | not started | —                                                                                                          |
| Stage 5 | not started | —                                                                                                          |
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
- Commit: `uncommitted`

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
- Commit: `uncommitted`

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
  - No run with a hosted embedding or chat model, and no ingestion of the real `corpus/` (deferred by request).
  - Ingestion retries (`max_retries`), deletion of documents removed from the corpus, and chunk-size checks against `EMBEDDING_MAX_INPUT_TOKENS` are not implemented.
  - Stage 1 still lacks pinned versions for several runtime dependencies.
- Commit: `uncommitted`
