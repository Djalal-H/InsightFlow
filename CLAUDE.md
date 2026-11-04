# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`AGENTS.md` is the authoritative project guide (conventions, guardrails, handoff rules); read it
first. This file adds commands and the architecture map.

## Communication

- Write all messages to the user in ASD-STE100 Simplified Technical English (writing rules).

## Commands

```bash
source .venv/bin/activate
ruff check src tests
mypy src                      # strict mode
pytest                        # coverage is on by default (addopts in pyproject.toml)
pytest tests/unit/test_max_min_chunker.py::test_name -p no:cacheprovider --no-cov   # single test
git diff --check              # required before handing off
QDRANT_TEST_URL=http://localhost:6333 pytest -m qdrant   # opt-in tests against a real server

docker compose up -d qdrant   # only when a workflow needs a real vector store
uvicorn insightflow.main:app --app-dir src --reload
PYTHONPATH=src python scripts/manual_chat_test.py "prompt"   # real provider call (billable)
```

`asyncio_mode = "auto"`, so async tests need no decorator. The default suite must run with no
credentials and no Qdrant; mock provider calls.

## Architecture

Modular monolith under `src/insightflow/`. Layers talk through protocols and the adapters wired in
`api/dependencies.py` (FastAPI `Depends`); LiteLLM and Qdrant types must not cross those boundaries.
Provider failures are mapped to `core.exceptions.ProviderError` and rendered by
`api/errors.py`.

### RAG pipeline (Stage 3, complete)

Flow: `DocumentSource` → `DocumentParser` → `NormalizedDocument` (ordered
`DocumentElement`s with page/heading provenance) → `Chunker` → `Chunk`s → (planned) embed → Qdrant.

- `rag/protocols.py` defines the boundaries (`DocumentParser`, `Chunker`, `ChunkContextualizer`,
  `Retriever`, `ContextAssembler`). Parsers/chunkers carry `*_name` and `*_version` attributes that
  feed into IDs, so bumping a version intentionally changes identities.
- `rag/identity.py` produces deterministic uuid5 IDs for documents, elements, and chunks from
  corpus-relative source paths and NFC-normalized content checksums. Preserve this determinism.
- `rag/parsers/`: `docling_pdf` (PDF), `docx`, `markdown`, `txt`, all emitting the same normalized model.
- `rag/chunkers/`: `structure_recursive` (structure-aware, sentence-split fallback, merges undersized
  chunks) and `max_min` (async semantic chunker that embeds sentences via `EmbeddingProvider`).
  Strategy and tuning come from `RAG_*` env vars via `rag/config.py` (`ChunkingConfig`).
- `rag/ingestion.py`: `IngestionPipeline` runs discovery → parser (selected by MIME type) →
  skip check → chunker → embedding → `ChunkStore.replace_document`. Only
  `DocumentRejectedError` becomes a per-document `rejected` result; other errors stop the run.
  `create_index_fingerprint` decides when an unchanged document is `skipped`.
- `storage/qdrant.py` implements `ChunkStore`; `chunk_to_payload`/`payload_to_chunk` define the
  point payload. Tests use `AsyncQdrantClient(location=":memory:")` instead of a server.
- `insightflow/ingest.py` is the CLI (`PYTHONPATH=src python -m insightflow.ingest <dir>`) and
  the place where settings become RAG configs.
- `rag/retrieval.py`: `DenseRetriever` (query embedding → `ChunkSearcher.search_chunks`) and
  `BudgetedContextAssembler` (rank-ordered passages within chunk/token limits, rendered as `[n]`).

### Query workflow (Stage 4)

`api/routes/query.py` serves `POST /query` and `POST /query/stream` (SSE, FastAPI's native
`fastapi.sse`). Both build the graph per request with `api/workflow.py:build_workflow` from
dependencies in `api/dependencies.py`, and map the final state with `to_query_response`.
The graph (`agents/graph.py`) is `route → (retrieve → answer → verify ⇄ rewrite | answer) →
finalize`; routing functions are pure and tested without a model. Nodes in `agents/nodes/` are
built by `make_*_node(deps)` factories; LLM nodes ask for JSON in the prompt and parse it with
`agents/structured.py`, falling back safely when the output is unusable.

- `AgentState` (`agents/state.py`): `history` persists across turns (append reducer); every
  other field is per-turn and must be reset with `new_turn_input`, because the checkpointer keeps
  the previous turn's values.
- Checkpointer (`agents/checkpoint.py`): opened once by the app lifespan in `main.py` and read by
  `get_checkpointer`. Pydantic types stored at the top level of the state must be listed in
  `CHECKPOINT_STATE_TYPES`, or LangGraph's serializer restores them as dicts.
- `GET/DELETE /threads/{id}` (`api/routes/threads.py`) read and delete checkpointed history.

API tests override `get_settings`, `get_chat_provider`, `get_retriever`, `get_context_assembler`,
and `get_checkpointer`; without these overrides the route opens a real Qdrant client.
`httpx.ASGITransport` does not run the lifespan, so requests are stateless unless
`get_checkpointer` is overridden. `tests/fakes.py:ScriptedChatProvider` replies per workflow role
(router, answer, judge, rewrite). LiteLLM loads `.env` into `os.environ` on import, so
`Settings(_env_file=None)` still sees `.env` values: pass values explicitly when a test depends
on them.

### Tools (Stage 5, in progress)

`tools/protocols.py:ToolClient` is the boundary for agent nodes; `tools/mcp_client.py:McpToolClient`
is the only MCP client code (`mcp` 2.x: `Client` and `MCPServer`; `FastMCP` was renamed).
`call_tool` never raises: allowlist, input-schema (jsonschema), timeout, server, and transport
failures become a `ToolResult` with an error code; oversized output is cut and marked `truncated`.
`list_tools` raises `ToolError`/`ToolTimeoutError`, and its result is cached per client instance. Each operation opens and closes its own MCP connection.
`tools/calculator_server.py` is an in-repo MCP server (tool `calculate`) over the `ast`-based
evaluator in `tools/calculator.py`. Tests connect to `MCPServer` instances in-process; one
integration test runs the server over stdio. `MCPServer()` calls `logging.basicConfig`, so
configure app logging before building an in-process server. The workflow does not call tools yet.

### Other layers

- `providers/`: `ChatProvider` / `EmbeddingProvider` protocols with LiteLLM adapters.
- `core/config.py`: pydantic-settings `Settings` (loaded from `.env`; see `.env.example`).
- `corpus/` holds sample DOCX/MD/PDF/TXT documents used for parser/chunker work.

## Documentation upkeep

`docs/progress/PROGRESS.md` is evidence-based: log validation commands exactly as run and do not
mark a stage complete from file presence alone. Roadmap and deferrals live in
`docs/Implement Plan Draft.md`.
