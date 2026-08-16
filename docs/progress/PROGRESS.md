# InsightFlow Progress

## Current status

- Current stage: Stage 3
- Last updated: 2026-08-14
- Overall state: in progress

## Stage summary

| Stage   | Status      | Evidence                                                                                                    |
| ------- | ----------- | ----------------------------------------------------------------------------------------------------------- |
| Stage 1 | in progress | Repository foundation is present, but tests, runtime dependency pins, and mypy validation remain incomplete |
| Stage 2 | complete    | Query workflow, provider validation and error mapping, and mocked contract coverage verified                |
| Stage 3 | in progress | PDF normalization and bounded Max-Min semantic chunking are implemented and tested                           |
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
