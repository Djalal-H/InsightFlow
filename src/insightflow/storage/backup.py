"""Backup, restore, and maintenance of the stores: Qdrant collections and graph checkpoints.

- Qdrant: a server-side collection snapshot, downloaded over the REST API and then removed
  from the server. Restore uploads the file with `priority=snapshot`, which replaces the
  collection's data. Snapshots need a Qdrant server; the in-memory client has none.
- Checkpoints (SQLite): the SQLite online backup API copies a consistent database while the
  application writes to it. A restore replaces the database file, so stop the application
  first.
- Redis holds only rate-limit counters that expire; it needs no backup.

A manifest records what a backup holds and which embedding model made the vectors, so a
restore into an incompatible configuration is refused.
"""

import hashlib
import json
import sqlite3
from collections.abc import Awaitable, Callable, Iterable
from contextlib import closing
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from importlib import metadata
from pathlib import Path
from typing import Any, Final

import httpx
from langgraph.checkpoint.base import BaseCheckpointSaver
from qdrant_client import AsyncQdrantClient, models

MANIFEST_NAME: Final = "manifest.json"
CHECKPOINTS_FILE: Final = "checkpoints.sqlite"
FORMAT_VERSION: Final = 1
DOWNLOAD_CHUNK_BYTES: Final = 1 << 20


@dataclass
class BackupManifest:
    """What one backup directory holds."""

    created_at: str
    app_version: str
    embedding_model: str
    embedding_dimensions: int | None
    retrieval_mode: str
    collections: list[dict[str, Any]] = field(default_factory=list)
    checkpoints: dict[str, Any] | None = None
    files: dict[str, str] = field(default_factory=dict)
    """File name → SHA-256, checked before a restore."""
    format_version: int = FORMAT_VERSION

    def write(self, directory: Path) -> Path:
        path = directory / MANIFEST_NAME
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", "utf-8")
        return path

    @classmethod
    def read(cls, directory: Path) -> "BackupManifest":
        data = json.loads((directory / MANIFEST_NAME).read_text("utf-8"))
        if data.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"unsupported backup format {data.get('format_version')!r}")
        return cls(**data)


class BackupError(Exception):
    """A backup or restore cannot continue; the message says why (no secrets in it)."""


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(DOWNLOAD_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def qdrant_http_client(
    url: str, api_key: str | None, *, timeout: float = 300.0
) -> httpx.AsyncClient:
    """An HTTP client for the snapshot endpoints, which qdrant-client does not wrap."""
    headers = {"api-key": api_key} if api_key else {}
    return httpx.AsyncClient(base_url=url.rstrip("/"), headers=headers, timeout=timeout)


async def backup_collection(
    client: AsyncQdrantClient, http: httpx.AsyncClient, collection: str, directory: Path
) -> dict[str, Any] | None:
    """Snapshot one collection into `directory`; `None` when the collection does not exist."""
    if not await client.collection_exists(collection):
        return None
    info = await client.get_collection(collection)
    snapshot = await client.create_snapshot(collection_name=collection, wait=True)
    if snapshot is None:
        raise BackupError(f"Qdrant made no snapshot of {collection}")
    target = directory / f"{collection}.snapshot"
    try:
        async with http.stream("GET", f"/collections/{collection}/snapshots/{snapshot.name}") as r:
            if r.status_code != 200:
                raise BackupError(f"snapshot download of {collection} failed ({r.status_code})")
            with target.open("wb") as handle:
                async for block in r.aiter_bytes(DOWNLOAD_CHUNK_BYTES):
                    handle.write(block)
    finally:
        await client.delete_snapshot(collection_name=collection, snapshot_name=snapshot.name)
    return {
        "name": collection,
        "file": target.name,
        "points": info.points_count,
        "vectors": info.config.params.model_dump(mode="json", exclude_none=True),
    }


async def restore_collection(
    http: httpx.AsyncClient, collection: str, snapshot_file: Path
) -> None:
    """Upload a snapshot; it replaces the collection's data (and creates the collection)."""
    with snapshot_file.open("rb") as handle:
        response = await http.post(
            f"/collections/{collection}/snapshots/upload",
            params={"priority": "snapshot", "wait": "true"},
            files={"snapshot": (snapshot_file.name, handle, "application/octet-stream")},
        )
    if response.status_code != 200:
        raise BackupError(f"snapshot upload of {collection} failed ({response.status_code})")


def backup_sqlite(source: Path, target: Path) -> None:
    """Copy a SQLite database consistently, even while another connection writes to it."""
    if not source.exists():
        raise BackupError(f"checkpoint database {source} does not exist")
    with (
        closing(sqlite3.connect(f"file:{source}?mode=ro", uri=True)) as src,
        closing(sqlite3.connect(target)) as dst,
    ):
        src.backup(dst)


def restore_sqlite(backup: Path, target: Path) -> None:
    """Replace the checkpoint database with a backup. Stop the application first."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(backup)) as src, closing(sqlite3.connect(target)) as dst:
        src.backup(dst)


def verify_files(manifest: BackupManifest, directory: Path) -> None:
    """Refuse a backup whose files are missing or changed."""
    for name, digest in manifest.files.items():
        path = directory / name
        if not path.exists():
            raise BackupError(f"backup file {name} is missing")
        if sha256_of(path) != digest:
            raise BackupError(f"backup file {name} does not match its checksum")


def compatibility_problems(
    manifest: BackupManifest, *, embedding_model: str, embedding_dimensions: int | None
) -> list[str]:
    """Why the backup's vectors do not fit the current configuration (empty: they fit)."""
    problems = []
    if manifest.embedding_model != embedding_model:
        problems.append("the embedding model differs from LITELLM_EMBEDDING_MODEL")
    if manifest.embedding_dimensions != embedding_dimensions:
        problems.append("the embedding dimensions differ from EMBEDDING_DIMENSIONS")
    return problems


async def switch_alias(client: AsyncQdrantClient, alias: str, collection: str) -> None:
    """Point `alias` at `collection` in one atomic change (re-index without downtime)."""
    if not await client.collection_exists(collection):
        raise BackupError(f"collection {collection} does not exist")
    current = await client.get_aliases()
    operations: list[Any] = []
    if any(item.alias_name == alias for item in current.aliases):
        operations.append(
            models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias))
        )
    operations.append(
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(collection_name=collection, alias_name=alias)
        )
    )
    await client.update_collection_aliases(change_aliases_operations=operations)


async def prune_threads(
    checkpointer: BaseCheckpointSaver[str],
    older_than: timedelta,
    *,
    now: datetime | None = None,
    forget: Callable[[str], Awaitable[None]] | None = None,
    dry_run: bool = False,
) -> list[str]:
    """Delete threads whose newest checkpoint is older than `older_than`; return their IDs.

    `forget` also removes a thread's conversation memory (before its checkpoints, so a failure
    leaves the thread in place and the prune can run again).
    """
    moment = now or datetime.now(UTC)
    newest: dict[str, datetime] = {}
    async for item in checkpointer.alist(None):
        thread_id = str(item.config.get("configurable", {}).get("thread_id", ""))
        stamp = _timestamp(item.checkpoint.get("ts"))
        if thread_id and stamp is not None and (
            thread_id not in newest or stamp > newest[thread_id]
        ):
            newest[thread_id] = stamp
    expired = sorted(thread for thread, stamp in newest.items() if moment - stamp > older_than)
    if not dry_run:
        for thread_id in expired:
            if forget is not None:
                await forget(thread_id)
            await checkpointer.adelete_thread(thread_id)
    return expired


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        stamp = datetime.fromisoformat(value)
    except ValueError:
        return None
    return stamp if stamp.tzinfo is not None else stamp.replace(tzinfo=UTC)


def checksums(directory: Path, names: Iterable[str]) -> dict[str, str]:
    return {name: sha256_of(directory / name) for name in names}
