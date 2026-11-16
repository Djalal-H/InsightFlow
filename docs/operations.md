# Operations

Procedures for a deployment of InsightFlow: images, secrets, backups, restores, re-indexing,
upgrades, retention, and load tests. The commands run from the repository root with the
virtual environment active, or in the `api` image (`python -m insightflow.ops ...`).

## What holds state

| Store | Contents | Backup |
| --- | --- | --- |
| Qdrant collection `QDRANT_COLLECTION` | Document chunks and vectors | Snapshot (`ops backup`) |
| Qdrant collection `MEMORY_COLLECTION` | Remembered conversation turns (only with `MEMORY_ENABLED`) | Snapshot (`ops backup`) |
| SQLite `CHECKPOINT_SQLITE_PATH` | Conversation threads: turns, owners, paused approvals | Online copy (`ops backup`) |
| Redis | Rate-limit and quota counters that expire | None: losing them resets the limits |

The corpus files are the source of the document collection: a full re-ingestion can always
rebuild it (`python -m insightflow.ingest`), but it calls the embedding model for every chunk.

## Images

```bash
docker build --target api -t insightflow-api .       # API, ~400 MB, no docling or torch
docker build --target ingest -t insightflow-ingest .  # ingestion CLI, CPU-only torch
docker run --rm -v "$PWD/corpus:/corpus:ro" --env-file .env insightflow-ingest /corpus
```

Both images run as user 10001 and write only to `/app/data` (checkpoints, model caches). The
API image works with a read-only root file system (`--read-only --tmpfs /tmp`). Set
`FORWARDED_ALLOW_IPS` to the load balancer's address so that Uvicorn trusts its
`X-Forwarded-*` headers. `compose.yaml` has an `api` service under the `app` profile:

```bash
mkdir -p secrets
openssl rand -hex 24 > secrets/redis_password
printf 'redis://:%s@redis:6379/0' "$(cat secrets/redis_password)" > secrets/redis_url
docker compose --profile app up -d --build
```

## Secrets

Every credential is a `SecretStr` setting and is redacted from logs (`Settings.secret_values()`
feeds the log formatter and the answer guard). In a container, mount the secrets as files and
set `SECRETS_DIR` (for example `/run/secrets`): a file named after a setting gives its value,
for example `/run/secrets/litellm_api_key`. Environment variables and `.env` win over the files.

Rotation: write the new value, then restart the API (settings are read once per process). For
`AUTH_JWKS_URL`, key rotation needs no restart: a token with an unknown `kid` makes the JWKS
client fetch the key set again (it caches keys for five minutes).

## Backup

```bash
PYTHONPATH=src python -m insightflow.ops backup --out backups/$(date -u +%Y%m%dT%H%M%SZ)
```

The command makes a server-side snapshot of each existing collection, downloads it, deletes it
from the server, copies the SQLite checkpoints with the SQLite online backup API (safe while the
API writes), and writes `manifest.json` with the embedding model, the dimensions, the vector
configuration, the package versions, and a SHA-256 of every file. Run it on a schedule and copy
the directory to storage outside the host. The output directory must not exist.

## Restore drill

Do a restore drill into a separate Qdrant and data directory at least once per release:

```bash
# 1. Stop the API (a checkpoint restore replaces the database file).
# 2. Restore; the manifest is checked against LITELLM_EMBEDDING_MODEL and EMBEDDING_DIMENSIONS.
PYTHONPATH=src python -m insightflow.ops restore --from backups/<stamp> --checkpoints
# 3. Start the API and check it.
curl -fsS localhost:8000/health/ready
PYTHONPATH=src python -m insightflow.evaluate --variants dense   # billable; compare with the baseline
```

A restore refuses changed files (checksums) and vectors from another embedding model or size;
`--force` skips only the model check. Without `--checkpoints`, only the collections are restored.

## Re-indexing without downtime (collection alias)

A change of the embedding model, the dimensions, the chunker version, or the retrieval mode
(dense to hybrid) needs a new collection. Let the API read an alias, and build the new
collection beside the old one:

```bash
# Once: point an alias at the current collection and set QDRANT_COLLECTION=<alias> for the API.
PYTHONPATH=src python -m insightflow.ops switch-alias --alias insightflow_docs --collection insightflow_documents
# Ingest into a new collection (the ingestion CLI writes the real collection name).
QDRANT_COLLECTION=insightflow_documents_v2 PYTHONPATH=src python -m insightflow.ingest corpus
# Compare retrieval quality on the new collection (billable), then switch atomically.
QDRANT_COLLECTION=insightflow_documents_v2 PYTHONPATH=src python -m insightflow.evaluate --baseline evaluation/baselines/<file>
PYTHONPATH=src python -m insightflow.ops switch-alias --alias insightflow_docs --collection insightflow_documents_v2
```

Keep the old collection until the new one has served traffic; switch back with the same
command if needed.

## Checkpoint upgrades and state changes

- Back up before a change of `langgraph` or `langgraph-checkpoint-sqlite`. `AsyncSqliteSaver.setup()`
  runs the package's own schema migrations when the API starts.
- Changes to `AgentState` must stay compatible with stored threads: add fields, never rename or
  remove them, and treat a missing field as its old meaning. Example: threads from before
  Stage 8 have no `owner`; only an `admin` token can open them.
- Pydantic types stored at the top level of the state must be listed in
  `CHECKPOINT_STATE_TYPES` (`agents/checkpoint.py`).
- A paused approval is stored in the checkpoint. It survives a restart; it ends when the user
  decides it or sends a new question on the thread.

## Retention

```bash
PYTHONPATH=src python -m insightflow.ops prune-threads --dry-run          # list only
PYTHONPATH=src python -m insightflow.ops prune-threads --older-than 30    # default: THREAD_RETENTION_DAYS
```

A thread whose newest checkpoint is older than the limit is deleted, with its remembered turns
when `MEMORY_ENABLED`. Run it daily.

## Load tests

```bash
PYTHONPATH=src python scripts/load_test.py --in-process --clients 20 --requests 400
PYTHONPATH=src python scripts/load_test.py --in-process --clients 10 --requests 200 --rate-limit 60 --burst 5
PYTHONPATH=src python scripts/load_test.py --url https://api.example --token "$TOKEN" --clients 5 --requests 50  # billable
```

The in-process mode uses a scripted model with simulated latency, so it measures the
application's overhead and its limits, not the provider. Results are in
`docs/progress/PROGRESS.md`.

## Failure behavior

| Failure | Response |
| --- | --- |
| Provider timeout / rate limit / error / configuration | 504 / 429 / 502 / 503 with a fixed code; LiteLLM retries transient errors `PROVIDER_MAX_RETRIES` times first |
| Provider fails during a stream | `error` event after the tokens sent so far; no `final` event |
| Qdrant unavailable | 503 `storage_unavailable` |
| Redis unavailable | 503 `rate_limiter_unavailable`, or admitted with `RATE_LIMIT_FAIL_OPEN=true` |
| Tool server down | The step records `timeout`/`unavailable`; the server pauses for `TOOL_FAILURE_COOLDOWN_SECONDS` |
| Conversation memory down | Logged; the turn continues without memory |

A failed turn does not change the thread: the next turn starts from the last good checkpoint
(`tests/integration/test_failure_simulation.py`).
