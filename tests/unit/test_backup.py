"""Backup and restore of checkpoints and Qdrant snapshots, alias switches, and pruning."""

import os
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TypedDict

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from qdrant_client import AsyncQdrantClient, models

from insightflow.agents.checkpoint import open_checkpointer, thread_config
from insightflow.ops import build_parser
from insightflow.storage.backup import (
    BackupError,
    BackupManifest,
    backup_collection,
    backup_sqlite,
    checksums,
    compatibility_problems,
    prune_threads,
    restore_collection,
    restore_sqlite,
    switch_alias,
    verify_files,
)


class Counter(TypedDict, total=False):
    n: int


def tiny_graph(checkpointer: Any) -> Any:
    async def step(state: Counter) -> Counter:
        return {"n": state.get("n", 0) + 1}

    graph = StateGraph(Counter)
    graph.add_node("step", step)
    graph.add_edge(START, "step")
    graph.add_edge("step", END)
    return graph.compile(checkpointer=checkpointer)


async def test_a_live_checkpoint_database_is_backed_up_and_restored(tmp_path: Path) -> None:
    live = tmp_path / "data" / "checkpoints.sqlite"
    copy = tmp_path / "backup.sqlite"
    restored = tmp_path / "restored" / "checkpoints.sqlite"
    async with open_checkpointer("sqlite", live) as saver:
        await tiny_graph(saver).ainvoke({"n": 1}, thread_config("t1"))
        backup_sqlite(live, copy)  # while the application's connection is open

    restore_sqlite(copy, restored)
    async with open_checkpointer("sqlite", restored) as saver:
        assert saver is not None
        stored = await saver.aget_tuple(thread_config("t1"))
    assert stored is not None and stored.checkpoint["channel_values"]["n"] == 2

    with pytest.raises(BackupError):
        backup_sqlite(tmp_path / "missing.sqlite", copy)


def manifest(**values: Any) -> BackupManifest:
    defaults: dict[str, Any] = {
        "created_at": "2026-11-15T00:00:00+00:00",
        "app_version": "0.1.0",
        "embedding_model": "nvidia_nim/embed",
        "embedding_dimensions": 1024,
        "retrieval_mode": "dense",
    }
    return BackupManifest(**{**defaults, **values})


def test_the_manifest_round_trips_and_checks_files(tmp_path: Path) -> None:
    (tmp_path / "a.snapshot").write_bytes(b"snapshot")
    written = manifest(files=checksums(tmp_path, ["a.snapshot"]))
    written.write(tmp_path)
    loaded = BackupManifest.read(tmp_path)
    assert loaded == written
    verify_files(loaded, tmp_path)

    (tmp_path / "a.snapshot").write_bytes(b"changed")
    with pytest.raises(BackupError, match="checksum"):
        verify_files(loaded, tmp_path)
    (tmp_path / "a.snapshot").unlink()
    with pytest.raises(BackupError, match="missing"):
        verify_files(loaded, tmp_path)


def test_an_unknown_backup_format_is_refused(tmp_path: Path) -> None:
    manifest(format_version=99).write(tmp_path)
    with pytest.raises(ValueError, match="format"):
        BackupManifest.read(tmp_path)


def test_incompatible_vectors_are_reported() -> None:
    backup = manifest()
    same = compatibility_problems(
        backup, embedding_model="nvidia_nim/embed", embedding_dimensions=1024
    )
    assert same == []
    problems = compatibility_problems(backup, embedding_model="openai/x", embedding_dimensions=1536)
    assert len(problems) == 2


class FakeSnapshotClient:
    """The qdrant-client calls that `backup_collection` makes."""

    def __init__(self) -> None:
        self.deleted: list[str] = []

    async def collection_exists(self, name: str) -> bool:
        return name == "docs"

    async def get_collection(self, name: str) -> Any:
        params = models.CollectionParams(
            vectors={"dense": models.VectorParams(size=4, distance=models.Distance.COSINE)}
        )
        config = models.CollectionConfig(
            params=params,
            hnsw_config=models.HnswConfig(m=16, ef_construct=100, full_scan_threshold=10000),
            optimizer_config=models.OptimizersConfig(
                deleted_threshold=0.2,
                vacuum_min_vector_number=1000,
                default_segment_number=0,
                flush_interval_sec=5,
            ),
            wal_config=models.WalConfig(wal_capacity_mb=32, wal_segments_ahead=0),
        )
        return models.CollectionInfo(
            status=models.CollectionStatus.GREEN,
            optimizer_status=models.OptimizersStatusOneOf.OK,
            points_count=3,
            segments_count=1,
            config=config,
            payload_schema={},
        )

    async def create_snapshot(self, collection_name: str, wait: bool) -> Any:
        return models.SnapshotDescription(name="docs-1.snapshot", size=8)

    async def delete_snapshot(self, collection_name: str, snapshot_name: str) -> None:
        self.deleted.append(snapshot_name)


async def test_a_collection_snapshot_is_downloaded_and_removed_from_the_server(
    tmp_path: Path,
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=b"SNAPSHOT")

    http = httpx.AsyncClient(
        base_url="http://qdrant.test",
        headers={"api-key": "k"},
        transport=httpx.MockTransport(handler),
    )
    client = FakeSnapshotClient()
    entry = await backup_collection(client, http, "docs", tmp_path)  # type: ignore[arg-type]
    missing = await backup_collection(client, http, "other", tmp_path)  # type: ignore[arg-type]

    assert entry is not None and missing is None
    assert (entry["file"], entry["points"]) == ("docs.snapshot", 3)
    assert entry["vectors"]["vectors"]["dense"]["size"] == 4
    assert (tmp_path / "docs.snapshot").read_bytes() == b"SNAPSHOT"
    assert seen[0].url.path == "/collections/docs/snapshots/docs-1.snapshot"
    assert seen[0].headers["api-key"] == "k"
    assert client.deleted == ["docs-1.snapshot"]


async def test_a_snapshot_upload_replaces_the_collection(tmp_path: Path) -> None:
    snapshot = tmp_path / "docs.snapshot"
    snapshot.write_bytes(b"SNAPSHOT")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200 if len(seen) == 1 else 500, json={})

    http = httpx.AsyncClient(base_url="http://qdrant.test", transport=httpx.MockTransport(handler))
    await restore_collection(http, "docs", snapshot)
    assert seen[0].url.path == "/collections/docs/snapshots/upload"
    assert seen[0].url.params["priority"] == "snapshot"
    assert b"SNAPSHOT" in seen[0].content
    with pytest.raises(BackupError, match="500"):
        await restore_collection(http, "docs", snapshot)


async def test_an_alias_moves_to_the_new_collection_atomically() -> None:
    client = AsyncQdrantClient(location=":memory:")
    for name in ("docs_v1", "docs_v2"):
        await client.create_collection(
            name, vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE)
        )
    await switch_alias(client, "docs", "docs_v1")
    await switch_alias(client, "docs", "docs_v2")
    aliases = (await client.get_aliases()).aliases
    assert [(item.alias_name, item.collection_name) for item in aliases] == [("docs", "docs_v2")]
    with pytest.raises(BackupError):
        await switch_alias(client, "docs", "missing")
    await client.close()


async def test_old_threads_are_pruned_with_their_memory() -> None:
    saver = InMemorySaver()
    graph = tiny_graph(saver)
    for thread in ("old", "new"):
        await graph.ainvoke({"n": 0}, thread_config(thread))
    forgotten: list[str] = []

    async def forget(thread_id: str) -> None:
        forgotten.append(thread_id)

    now = datetime.now(UTC)
    assert await prune_threads(saver, timedelta(days=30), now=now) == []
    later = now + timedelta(days=31)
    assert await prune_threads(saver, timedelta(days=30), now=later, dry_run=True) == [
        "new",
        "old",
    ]
    assert await saver.aget_tuple(thread_config("old")) is not None
    assert await prune_threads(saver, timedelta(days=30), now=later, forget=forget) == [
        "new",
        "old",
    ]
    assert forgotten == ["new", "old"]
    assert await saver.aget_tuple(thread_config("old")) is None


def test_the_cli_parses_every_command() -> None:
    parser = build_parser()
    assert parser.parse_args(["backup", "--out", "b"]).out == Path("b")
    restore = parser.parse_args(["restore", "--from", "b", "--checkpoints"])
    assert (restore.source, restore.checkpoints, restore.force) == (Path("b"), True, False)
    assert parser.parse_args(["switch-alias", "--alias", "a", "--collection", "c"]).alias == "a"
    assert parser.parse_args(["prune-threads", "--dry-run"]).older_than is None


@pytest.mark.qdrant
async def test_a_snapshot_round_trip_on_a_real_qdrant(tmp_path: Path) -> None:
    url = os.environ.get("QDRANT_TEST_URL")
    if not url:
        pytest.skip("QDRANT_TEST_URL is not set")
    from insightflow.storage.backup import qdrant_http_client

    name = f"backup_test_{uuid.uuid4().hex[:8]}"
    client = AsyncQdrantClient(url=url)
    http = qdrant_http_client(url, None)
    try:
        await client.create_collection(
            name, vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE)
        )
        await client.upsert(
            name, points=[models.PointStruct(id=1, vector=[0.1, 0.2], payload={"k": "v"})]
        )
        entry = await backup_collection(client, http, name, tmp_path)
        assert entry is not None and entry["points"] == 1
        await client.delete_collection(name)
        await restore_collection(http, name, tmp_path / entry["file"])
        [point] = (await client.retrieve(name, ids=[1]))
        assert point.payload == {"k": "v"}
    finally:
        await client.delete_collection(name)
        await http.aclose()
        await client.close()
