# InsightFlow Roadmap Reference

## Architecture constraints

- Use hosted chat, embedding, and reranking APIs only.
- Use LiteLLM behind provider-independent contracts.
- Use Qdrant as the vector store.
- Start as a modular monolith.
- Keep FastAPI transport schemas, LangGraph state, provider adapters, RAG logic, and storage payloads separated.
- Add advanced infrastructure only when requirements justify it.

## Stage 1 — Repository and development foundation

Evidence includes modular `src/insightflow` boundaries, FastAPI application factory, health endpoints, provider contracts, LiteLLM adapters, Qdrant adapter, Pydantic settings, pinned dependencies, `.env.example`, local virtual environment guidance, Qdrant-only Compose, tests, README, and architecture documentation.

Do not treat application Dockerfiles, Redis, MCP, authentication, large observability stacks, or CI/CD as required Stage 1 work.

## Stage 2 — Minimal hosted-model workflow

Evidence includes request/response schemas, a non-streaming query endpoint, a minimal LangGraph node calling the chat-provider contract, provider configuration validation, stable provider error mapping, mocked unit tests, and optional real-provider tests.

## Stage 3 — Basic ingestion and RAG

Evidence includes supported format selection, normalization, traceable chunking, hosted embeddings, Qdrant collection setup, point payload mapping, dense retrieval, grounded generation, structured answers, source identifiers, persistence verification, and tests.

## Stage 4 — Workflow expansion and persistence

Evidence includes explicit planner/router, retriever, answer, and verifier nodes; bounded conditional loops; structured execution metadata; and persistent checkpointing after resumability requirements are defined.

## Stage 5 — MCP tool integration

Evidence includes the official MCP SDK, a dedicated client boundary, one bounded low-risk tool, schemas, timeouts, output limits, audit metadata, and routing/termination tests.

## Stage 6 — Retrieval quality and memory

Evidence must be tied to measured retrieval failures and comparison on a labeled evaluation set. Candidate work includes query rewriting, metadata filtering, hybrid retrieval, reranking, context compression, multi-hop retrieval, and retrieval-based memory.

## Stage 7 — Observability, evaluation, and reliability

Evidence includes correlated structured logs, latency and token metrics, retrieval and retry metrics, deterministic regression datasets, and opt-in budgeted API evaluation. OpenTelemetry, Langfuse, Prometheus, and Grafana are candidates, not automatic requirements.

## Stage 8 — Security and production readiness

Evidence includes authentication, authorization, rate limiting, quotas, secret management, sensitive-log controls, prompt-injection defenses, tool authorization, application containerization, CI checks, backup/migration procedures, load tests, and provider-failure simulations.
