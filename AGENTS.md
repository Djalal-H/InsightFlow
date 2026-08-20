# AGENTS.md

## Project overview

InsightFlow is a Python 3.12–3.13 modular monolith for API-first agent orchestration and
retrieval-augmented generation. The application uses FastAPI, LangGraph, LiteLLM, and Qdrant.
Keep external providers and storage behind the repository's protocols and adapters.

Read these documents before making architectural changes:

- [`README.md`](README.md) — setup, development commands, and repository structure.
- [`docs/Project Overview.md`](docs/Project%20Overview.md) — architecture principles and boundaries.
- [`docs/Implement Plan Draft.md`](docs/Implement%20Plan%20Draft.md) — staged roadmap and explicit deferrals.
- [`docs/progress/PROGRESS.md`](docs/progress/PROGRESS.md) — verified implementation status.

## Repository layout

- `src/insightflow/api/` — FastAPI routes, dependencies, and transport schemas.
- `src/insightflow/agents/` — LangGraph state, graph construction, and nodes.
- `src/insightflow/providers/` — provider-neutral chat/embedding contracts and LiteLLM adapters.
- `src/insightflow/rag/` — ingestion, parsers, chunking, retrieval, and RAG domain models.
- `src/insightflow/storage/` — Qdrant integration.
- `src/insightflow/core/` — settings, logging, and shared exceptions.
- `tests/unit/` — deterministic unit and contract tests.
- `tests/integration/` — API and external-boundary tests; keep real services opt-in.

## Development workflow

Use the repository virtual environment when available. From the repository root:

```bash
source .venv/bin/activate
ruff check src tests
mypy src
pytest
```

Run Qdrant locally only when a test or manual workflow needs it:

```bash
docker compose up -d qdrant
uvicorn insightflow.main:app --app-dir src --reload
```

The default test suite must remain deterministic and must not require hosted-model credentials or
running Qdrant. Mock provider calls in unit tests. Mark or isolate tests that require real services
and never make them part of the default developer workflow without updating the documentation.

## Coding conventions

- Target Python 3.12 syntax and typing; keep Ruff's configured 100-character line length.
- Run Ruff, mypy, and pytest for code changes. Run `git diff --check` before handing off changes.
- Preserve strict mypy compatibility and add types at public boundaries.
- Prefer small, explicit modules and dependency injection over global clients or hidden state.
- Keep FastAPI schemas, LangGraph state, domain models, provider contracts, and storage payloads
  separate. Do not leak LiteLLM or Qdrant-specific objects across those boundaries.
- Map provider failures to stable, sanitized application errors. Never expose credentials, raw
  provider details, or sensitive prompts/responses in API responses or logs.
- Give ingested content deterministic, traceable identities and preserve source/page metadata.
- Add or update focused tests with every behavior change, especially for public contracts and
  adapter translations.

## Scope and architecture guardrails

- Hosted model APIs are the active model strategy. Do not add local LLM, embedding, reranking,
  GPU, or model-download dependencies without an explicit architecture decision.
- Qdrant is the vector-store boundary. Configure connection details and model identifiers through
  settings; do not hard-code environment-specific values.
- Keep the project a modular monolith until a demonstrated scaling or ownership need justifies a
  service split.
- MCP, authentication, observability stacks, persistent checkpoints, application containerization,
  and production infrastructure are roadmap work, not assumptions for routine changes.
- Add dependencies deliberately to the appropriate requirements file and update documentation when
  setup or runtime behavior changes.
- Never commit `.env`, API keys, generated data, caches, or local Qdrant storage.

## Documentation and handoff

Update the relevant README or `docs/` page when a change alters architecture, configuration, public
API behavior, or the roadmap. Keep `docs/progress/PROGRESS.md` evidence-based: record validation
commands exactly as run, distinguish mocked from real integrations, and do not mark roadmap stages
complete from file presence alone.

Before finishing, summarize changed files, validation results, and any deferred or unverified work.
