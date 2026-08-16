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
in unit tests; real credentials belong only in opt-in integration tests.

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

For manual inspection with real sentence embeddings, open
`notebooks/inspect_max_min_chunking.ipynb`, select the repository `.venv` kernel, set
`DOCUMENT_PATH`, and run the cells in order. The notebook reads model credentials from `.env`,
prints complete chunk provenance and previews, and performs no Qdrant writes. Its embedding cell
makes billable provider requests; the optional JSON export remains disabled until `OUTPUT_PATH` is
set explicitly. Clear executed notebook outputs before committing to avoid retaining document
content in Git.
