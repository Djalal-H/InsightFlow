# syntax=docker/dockerfile:1.7
# Two images from one file:
#   docker build --target api    -t insightflow-api .     # the HTTP API (no docling, no torch)
#   docker build --target ingest -t insightflow-ingest .  # the ingestion CLI (CPU-only torch)
# Dependencies come from the lock files; regenerate them with the commands in README.md.

ARG PYTHON_IMAGE=python:3.13.12-slim-bookworm
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.7.3

FROM ${UV_IMAGE} AS uv

FROM ${PYTHON_IMAGE} AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH
WORKDIR /app

FROM base AS api-deps
COPY --from=uv /uv /usr/local/bin/uv
COPY requirements.lock ./
RUN uv venv /opt/venv && uv pip install --no-cache -r requirements.lock

FROM base AS ingest-deps
COPY --from=uv /uv /usr/local/bin/uv
COPY requirements-ingest.lock ./
RUN uv venv /opt/venv && uv pip install --no-cache --torch-backend cpu -r requirements-ingest.lock

FROM base AS runtime
# A fixed non-root user; /app/data is the only directory that it writes to.
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app --no-create-home app \
    && mkdir -p /app/data \
    && chown app:app /app/data
ENV PYTHONPATH=/app/src \
    # LiteLLM reads its bundled model registry instead of downloading one at import.
    LITELLM_LOCAL_MODEL_COST_MAP=True \
    CHECKPOINT_SQLITE_PATH=/app/data/checkpoints.sqlite \
    HF_HOME=/app/data/.cache/huggingface

FROM runtime AS api
COPY --from=api-deps /opt/venv /opt/venv
COPY src ./src
USER app
VOLUME ["/app/data"]
EXPOSE 8000
# Proxies whose X-Forwarded-* headers Uvicorn trusts; set it to your load balancer's address.
ENV FORWARDED_ALLOW_IPS=127.0.0.1
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=4).status != 200)"]
CMD ["sh", "-c", "exec uvicorn insightflow.main:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips \"$FORWARDED_ALLOW_IPS\" --no-server-header --no-access-log"]

FROM runtime AS ingest
COPY --from=ingest-deps /opt/venv /opt/venv
COPY src ./src
USER app
VOLUME ["/app/data"]
ENTRYPOINT ["python", "-m", "insightflow.ingest"]
