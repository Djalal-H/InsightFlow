"""Operations CLI: backup, restore, alias switch, and thread retention.

    PYTHONPATH=src python -m insightflow.ops backup --out backups/2026-11-15
    PYTHONPATH=src python -m insightflow.ops restore --from backups/2026-11-15 [--checkpoints]
    PYTHONPATH=src python -m insightflow.ops switch-alias --alias docs --collection docs_v2
    PYTHONPATH=src python -m insightflow.ops prune-threads [--older-than 30] [--dry-run]

Each command prints one JSON summary. See `docs/operations.md` for the procedures.
"""

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from qdrant_client import AsyncQdrantClient

from insightflow.agents.checkpoint import open_checkpointer
from insightflow.core.config import Settings, get_settings, secret_text
from insightflow.storage.backup import (
    CHECKPOINTS_FILE,
    BackupError,
    BackupManifest,
    backup_collection,
    backup_sqlite,
    checksums,
    compatibility_problems,
    package_version,
    prune_threads,
    qdrant_http_client,
    restore_collection,
    restore_sqlite,
    switch_alias,
    verify_files,
)
from insightflow.storage.qdrant_memory import QdrantMemoryStore


def qdrant_client(settings: Settings) -> AsyncQdrantClient:
    return AsyncQdrantClient(url=settings.qdrant_url, api_key=secret_text(settings.qdrant_api_key))


async def backup(settings: Settings, out: Path) -> dict[str, Any]:
    """Write a backup into `out`, a directory that the caller made empty."""
    manifest = BackupManifest(
        created_at=datetime.now(UTC).isoformat(),
        app_version=package_version("insightflow"),
        embedding_model=settings.litellm_embedding_model,
        embedding_dimensions=settings.embedding_dimensions,
        retrieval_mode=settings.rag_retrieval_mode,
    )
    client = qdrant_client(settings)
    http = qdrant_http_client(settings.qdrant_url, secret_text(settings.qdrant_api_key))
    try:
        for collection in (settings.qdrant_collection, settings.memory_collection):
            entry = await backup_collection(client, http, collection, out)
            if entry is not None:
                manifest.collections.append(entry)
    finally:
        await http.aclose()
        await client.close()
    if settings.checkpoint_backend == "sqlite":
        backup_sqlite(settings.checkpoint_sqlite_path, out / CHECKPOINTS_FILE)
        manifest.checkpoints = {
            "backend": "sqlite",
            "file": CHECKPOINTS_FILE,
            "package": "langgraph-checkpoint-sqlite "
            + package_version("langgraph-checkpoint-sqlite"),
        }
    names = [entry["file"] for entry in manifest.collections]
    if manifest.checkpoints is not None:
        names.append(CHECKPOINTS_FILE)
    manifest.files = checksums(out, names)
    manifest.write(out)
    return {"backup": str(out), "collections": manifest.collections, "files": manifest.files}


async def restore(
    settings: Settings, source: Path, *, checkpoints: bool, force: bool
) -> dict[str, Any]:
    manifest = BackupManifest.read(source)
    verify_files(manifest, source)
    problems = compatibility_problems(
        manifest,
        embedding_model=settings.litellm_embedding_model,
        embedding_dimensions=settings.embedding_dimensions,
    )
    if problems and not force:
        raise BackupError("refused: " + "; ".join(problems) + " (use --force to restore anyway)")
    http = qdrant_http_client(settings.qdrant_url, secret_text(settings.qdrant_api_key))
    try:
        for entry in manifest.collections:
            await restore_collection(http, entry["name"], source / entry["file"])
    finally:
        await http.aclose()
    restored_checkpoints = False
    if checkpoints and manifest.checkpoints is not None:
        restore_sqlite(source / manifest.checkpoints["file"], settings.checkpoint_sqlite_path)
        restored_checkpoints = True
    return {
        "restored_collections": [entry["name"] for entry in manifest.collections],
        "restored_checkpoints": restored_checkpoints,
        "warnings": problems,
    }


async def alias(settings: Settings, name: str, collection: str) -> dict[str, Any]:
    client = qdrant_client(settings)
    try:
        await switch_alias(client, name, collection)
    finally:
        await client.close()
    return {"alias": name, "collection": collection}


async def prune(settings: Settings, days: int, *, dry_run: bool) -> dict[str, Any]:
    memory = QdrantMemoryStore(settings) if settings.memory_enabled else None
    try:
        async with open_checkpointer(
            settings.checkpoint_backend, settings.checkpoint_sqlite_path
        ) as checkpointer:
            if checkpointer is None:
                return {"pruned": [], "dry_run": dry_run}
            pruned = await prune_threads(
                checkpointer,
                timedelta(days=days),
                forget=memory.delete_thread if memory is not None else None,
                dry_run=dry_run,
            )
    finally:
        if memory is not None:
            await memory.close()
    return {"pruned": pruned, "dry_run": dry_run, "older_than_days": days}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="InsightFlow store operations.")
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("backup", help="Snapshot Qdrant and copy the checkpoints.")
    make.add_argument("--out", type=Path, required=True, help="A new directory for the backup.")
    load = commands.add_parser("restore", help="Restore a backup directory.")
    load.add_argument("--from", dest="source", type=Path, required=True)
    load.add_argument(
        "--checkpoints",
        action="store_true",
        help="Also replace the checkpoint database (stop the application first).",
    )
    load.add_argument("--force", action="store_true", help="Ignore the compatibility check.")
    swap = commands.add_parser("switch-alias", help="Point a Qdrant alias at a collection.")
    swap.add_argument("--alias", required=True)
    swap.add_argument("--collection", required=True)
    clean = commands.add_parser("prune-threads", help="Delete threads older than the retention.")
    clean.add_argument("--older-than", type=int, default=None, help="Days (THREAD_RETENTION_DAYS).")
    clean.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    try:
        if args.command == "backup":
            args.out.mkdir(parents=True, exist_ok=False)
            result = asyncio.run(backup(settings, args.out))
        elif args.command == "restore":
            result = asyncio.run(
                restore(settings, args.source, checkpoints=args.checkpoints, force=args.force)
            )
        elif args.command == "switch-alias":
            result = asyncio.run(alias(settings, args.alias, args.collection))
        else:
            days = args.older_than or settings.thread_retention_days
            result = asyncio.run(prune(settings, days, dry_run=args.dry_run))
    except (BackupError, ValueError, OSError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
