# Project Overview: InsightFlow

InsightFlow is an open-source, API-first AI workflow orchestrator that combines stateful agent
workflows, retrieval-augmented generation (RAG), and tool invocation. It is designed as a serious
AI engineering project rather than a single-turn chatbot: a request can be planned, enriched with
retrieved knowledge, routed to tools, verified, and returned with sources and execution metadata.

This document describes the intended architecture. The implementation plan remains flexible and
features should be introduced only when the preceding end-to-end workflow is stable.

## Architecture principles

1. **Hosted model APIs only.** InsightFlow does not host language, embedding, or reranking models.
   Chat and embedding requests go through LiteLLM so providers and models can change through
   configuration without changing agent or RAG code.
2. **Qdrant is the vector store.** Qdrant stores document chunks, externally generated vectors,
   metadata, and—when needed later—retrieval-based memory.
3. **Start as a modular monolith.** The initial FastAPI, LangGraph, provider, RAG, and storage code
   lives in one Python package. Components can become services only when scaling or ownership
   requirements justify that cost.
4. **Build incrementally.** The first useful slice is one API workflow with retrieval and a grounded
   response. Multi-agent collaboration, MCP, advanced retrieval, and production infrastructure are
   later additions.
5. **Keep external systems behind interfaces.** Agent nodes depend on chat, embedding, retrieval,
   and tool contracts rather than vendor SDKs or database clients directly.

## System architecture

A client sends a request to the FastAPI application. LangGraph coordinates explicit workflow nodes
and keeps request state as it moves through planning, retrieval, generation, and verification. Chat
and embedding nodes call hosted APIs through the in-process LiteLLM adapter. Retrieval nodes query
Qdrant through a storage adapter. Later, tool nodes can call MCP servers through a dedicated client
boundary.

The initial deployment has only two runtime processes:

- The Python application, run directly in a dedicated virtual environment.
- Qdrant, run locally with Docker Compose or replaced by Qdrant Cloud through configuration.

There is no local LLM server, GPU runtime, model download, or model container. A separately deployed
LiteLLM Proxy may be evaluated later if centralized routing, budgets, or cross-application policy
becomes necessary.

## Core components

### FastAPI application

FastAPI exposes health endpoints and, in the first functional milestone, query and ingestion APIs.
Transport schemas stay separate from LangGraph state and storage payloads. Streaming can be added to
the query API once the non-streaming path is reliable.

### LangGraph orchestration

LangGraph defines deterministic nodes and conditional transitions. The initial graph is deliberately
small; later workflows can introduce planner, retriever, executor, tool, and verifier nodes, retries,
human approval, subgraphs, and durable checkpoints.

### Hosted model providers

LiteLLM provides a uniform interface for hosted chat and embedding APIs. Configuration selects model
identifiers and the corresponding provider credentials. Agents consume a chat-provider contract and
RAG consumes an embedding-provider contract, keeping LiteLLM details at the integration boundary.

API-backed reranking or evaluation can be added later through similar interfaces. Local Transformers,
Ollama, SentenceTransformers, FastEmbed, and local cross-encoders are outside the active design.

### RAG and Qdrant

The ingestion pipeline will normalize supported input formats, chunk content, request embeddings from
the configured hosted API, and store each chunk as a Qdrant point with source metadata. Document
parsers are deferred until the first supported formats are selected.

The initial retrieval path is dense vector search with metadata filters. Hybrid dense/sparse search,
query rewriting, parent-child retrieval, reranking, context compression, multi-hop retrieval, and
memory are later improvements that should be justified with evaluation data.

### Tools and MCP

MCP remains the preferred future protocol for external tools and data sources. It is not part of the
initial dependency set or repository scaffold. The first MCP milestone should add one safe tool and
test tool selection, error handling, timeouts, and result integration before expanding the catalog.

### Observability, security, and persistence

Structured logs and basic request metrics should precede a larger observability stack. Langfuse,
OpenTelemetry, Prometheus, and Grafana are candidates for later milestones, not initial dependencies.
Authentication, authorization, rate limiting, prompt-injection controls, secrets management, and
persistent LangGraph checkpoints should be added when the relevant external interfaces exist.

## Repository structure

```text
.
├── src/insightflow/
│   ├── main.py
│   ├── api/                    # HTTP routes and transport schemas
│   ├── agents/                 # LangGraph state, graph, and nodes
│   ├── rag/                    # Ingestion and retrieval domain boundaries
│   ├── providers/              # LiteLLM chat and embedding adapters
│   ├── storage/                # Qdrant adapter
│   └── core/                   # Settings, logging, and shared errors
├── tests/
│   ├── unit/
│   └── integration/
├── docs/
├── compose.yaml                # Local Qdrant only
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── .env.example
└── README.md
```

Separate microservice directories, application containers, Kubernetes manifests, and infrastructure
as code should be introduced only when the modular monolith has a demonstrated limitation.

## Technology choices

- Python 3.12 or 3.13
- FastAPI and Uvicorn for the HTTP application
- LangGraph for stateful orchestration
- LiteLLM's Python SDK for hosted chat and embedding APIs
- Qdrant and `qdrant-client` for vector storage and retrieval
- Pydantic Settings for environment configuration
- Pytest, Ruff, and mypy for development checks
- Docker Compose for local Qdrant only

Provider keys belong in the local environment or a deployed secret manager and must never be committed.
Qdrant connection details and model identifiers are environment-driven so local and hosted services use
the same application interfaces.

## Target workflow

The first complete RAG workflow should follow this path:

1. Validate an incoming query.
2. Generate a query embedding through the configured LiteLLM model.
3. Retrieve relevant chunks and source metadata from Qdrant.
4. Generate a grounded answer through the configured chat API.
5. Return structured answer and source fields.

Later versions may add planning, query rewriting, tools, verification, retry loops, long-term memory,
and multiple specialized agents. Each additional node should have a measurable purpose and isolated
tests.

## Evaluation strategy

- End-to-end latency and provider token usage
- Retrieval precision/recall on a small held-out dataset
- Answer correctness and source coverage
- Unsupported-claim or hallucination rate
- Tool success rate when MCP is introduced
- Workflow completion and retry rates

Unit tests mock hosted model calls. Real-provider and Qdrant tests are opt-in integration tests so the
default suite remains deterministic and does not consume API budget.

## Production direction

After the MVP works, production hardening may include persistent checkpoints, request tracing, caching,
rate limits, authentication, audit logs, prompt and tool guardrails, CI/CD, an application image, and a
managed deployment. Kubernetes, Redis, a LiteLLM Proxy, and multiple application services are options,
not prerequisites.

