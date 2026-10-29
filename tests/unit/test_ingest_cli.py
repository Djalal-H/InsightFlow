"""Tests for the corpus ingestion command-line entry point."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from qdrant_client import AsyncQdrantClient

from insightflow import ingest
from insightflow.core.config import Settings
from insightflow.core.exceptions import ProviderConfigurationError, StorageError
from insightflow.rag.models import IngestionResult, IngestionWarning
from insightflow.storage.qdrant import QdrantStore


def make_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "embedding_provider": "openai",
        "litellm_embedding_model": "openai/text-embedding-3-small",
        "embedding_dimensions": 1536,
        "embedding_max_input_tokens": 8191,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


RESULTS = [
    IngestionResult(
        source_identifier="a.txt",
        status="completed",
        document_id="d1",
        document_version="v1",
        chunk_count=3,
    ),
    IngestionResult(
        source_identifier="scan.pdf",
        status="rejected",
        warnings=[IngestionWarning(code="scanned_pdf", message="Document rejected")],
    ),
]


def test_embedding_config_requires_vector_shape_settings() -> None:
    settings = make_settings(embedding_dimensions=None, embedding_provider="")

    with pytest.raises(ProviderConfigurationError) as error:
        ingest.embedding_config_from_settings(settings)

    assert str(error.value) == "Missing settings: EMBEDDING_PROVIDER, EMBEDDING_DIMENSIONS"


def test_embedding_config_uses_the_qdrant_distance() -> None:
    config = ingest.embedding_config_from_settings(make_settings(qdrant_distance="dot"))

    assert config.dimensions == 1536
    assert config.distance == "dot"


def test_main_prints_results_and_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with (
        patch.object(ingest, "get_settings", return_value=make_settings()),
        patch.object(ingest, "run", new_callable=AsyncMock, return_value=RESULTS) as run,
    ):
        exit_code = ingest.main([str(tmp_path), "--force"])

    assert exit_code == 0
    assert run.await_args is not None
    assert run.await_args.kwargs == {"force": True}
    output = capsys.readouterr().out.splitlines()
    assert output[0].split() == ["completed", "3", "a.txt"]
    assert output[1].split() == ["rejected", "0", "scan.pdf", "[scanned_pdf]"]
    assert output[2] == "2 documents: 1 completed, 1 rejected"


def test_main_reports_application_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with (
        patch.object(ingest, "get_settings", return_value=make_settings()),
        patch.object(
            ingest,
            "run",
            new_callable=AsyncMock,
            side_effect=StorageError("Qdrant collection setup failed"),
        ),
    ):
        exit_code = ingest.main([str(tmp_path)])

    assert exit_code == 1
    assert "StorageError: Qdrant collection setup failed" in capsys.readouterr().err


def test_main_requires_an_existing_corpus_directory(tmp_path: Path) -> None:
    with (
        patch.object(ingest, "get_settings", return_value=make_settings()),
        pytest.raises(SystemExit) as exit_info,
    ):
        ingest.main([str(tmp_path / "missing")])

    assert exit_info.value.code == 2


def test_main_uses_the_configured_corpus_directory(tmp_path: Path) -> None:
    with (
        patch.object(ingest, "get_settings", return_value=make_settings(rag_corpus_dir=tmp_path)),
        patch.object(ingest, "run", new_callable=AsyncMock, return_value=[]) as run,
    ):
        assert ingest.main([]) == 0

    assert run.await_args is not None
    assert run.await_args.args[0] == tmp_path


@pytest.mark.filterwarnings("ignore:Payload indexes have no effect")
async def test_run_composes_adapters_from_settings(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("A short document.", encoding="utf-8")
    embedder = AsyncMock()
    embedder.embed_documents.side_effect = lambda texts: [[1.0, 0.0, 0.0] for _ in texts]

    def in_memory_store(settings: Settings) -> QdrantStore:
        return QdrantStore(settings, client=AsyncQdrantClient(location=":memory:"))

    with (
        patch.object(ingest, "LiteLLMEmbeddingProvider", return_value=embedder),
        patch.object(ingest, "QdrantStore", side_effect=in_memory_store),
    ):
        results = await ingest.run(tmp_path, make_settings(embedding_dimensions=3), force=False)

    assert [(result.source_identifier, result.status) for result in results] == [
        ("a.txt", "completed")
    ]
