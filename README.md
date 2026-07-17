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
