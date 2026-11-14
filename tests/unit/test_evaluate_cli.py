"""End-to-end tests for the evaluation command on a small in-memory index."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import patch

import pytest
from qdrant_client import AsyncQdrantClient

from insightflow import evaluate
from insightflow.core.config import Settings
from insightflow.evaluation.retrieval import load_cases
from insightflow.rag.chunkers import StructureRecursiveChunker
from insightflow.rag.config import ChunkingConfig, EmbeddingConfig
from insightflow.rag.ingestion import IngestionPipeline
from insightflow.rag.parsers.txt import TxtParser
from insightflow.rag.sparse import Bm25SparseEncoder
from insightflow.storage.qdrant import QdrantStore
from tests.fakes import (
    HYBRID_DOCUMENTS,
    HYBRID_QUERY,
    KEYWORDS,
    KeywordEmbeddingProvider,
    ScriptedChatProvider,
)

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")

CASES = [
    {
        "id": "code",
        "question": HYBRID_QUERY,
        "relevant": [{"source": "cs303.txt", "evidence": ["Course CS303"]}],
        "tags": ["keyword"],
    },
    {
        "id": "peak",
        "question": "How tall is the mountain?",
        "relevant": [{"source": "peak.txt", "evidence": ["tall"]}],
        "tags": ["fact"],
    },
]


class SharedStore(QdrantStore):
    """The command closes its store; keep the in-memory client open for the next variant."""

    async def close(self) -> None:
        return None


SETTINGS = Settings(
    _env_file=None,
    qdrant_collection="test_evaluate",
    litellm_rerank_model="",
    rag_retrieval_mode="dense",
    rag_candidate_top_k=4,
    rag_generation_max_chunks=1,
)


@pytest.fixture
async def indexed_store(tmp_path: Path) -> AsyncIterator[SharedStore]:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name, text in HYBRID_DOCUMENTS.items():
        (corpus / name).write_text(text, encoding="utf-8")
    client = AsyncQdrantClient(location=":memory:")
    store = SharedStore(SETTINGS, client=client)
    await IngestionPipeline(
        parsers={"text/plain": TxtParser()},
        chunker=StructureRecursiveChunker(),
        chunking_config=ChunkingConfig(
            strategy="structure_recursive", target_tokens=100, max_tokens=200, min_tokens=1
        ),
        embedding_provider=KeywordEmbeddingProvider(),
        embedding_config=EmbeddingConfig(
            provider="fake", model="keyword", dimensions=len(KEYWORDS), max_input_tokens=512
        ),
        store=store,
        sparse_encoder=Bm25SparseEncoder(),
    ).ingest_corpus(corpus)
    yield store
    await client.close()


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(json.dumps(case) for case in CASES), encoding="utf-8")
    return path


async def test_run_compares_variants_on_the_same_index(
    indexed_store: SharedStore, dataset: Path
) -> None:
    with (
        patch.object(evaluate, "QdrantStore", return_value=indexed_store),
        patch.object(evaluate, "get_embedding_provider", return_value=KeywordEmbeddingProvider()),
        patch.object(evaluate, "get_chat_provider", return_value=ScriptedChatProvider()),
    ):
        dense, hybrid, rerank = await evaluate.run(
            load_cases(dataset), ["dense", "hybrid", "dense+rerank"], SETTINGS
        )

    # One passage enters the context. Dense search puts `rivers.txt` before the course.
    assert (dense.mean("hit", "keyword"), hybrid.mean("hit", "keyword")) == (0.0, 1.0)
    assert (dense.mean("hit", "fact"), hybrid.mean("hit", "fact")) == (1.0, 1.0)
    assert (dense.mean("hit"), hybrid.mean("hit")) == (0.5, 1.0)
    assert all(case.passages == 1 and case.context_tokens > 0 for case in hybrid.cases)
    assert rerank.error == "LITELLM_RERANK_MODEL is not set"


def test_main_prints_the_table_and_writes_json(
    indexed_store: SharedStore,
    dataset: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "scores.json"
    with (
        patch.object(evaluate, "get_settings", return_value=SETTINGS),
        patch.object(evaluate, "QdrantStore", return_value=indexed_store),
        patch.object(evaluate, "get_embedding_provider", return_value=KeywordEmbeddingProvider()),
        patch.object(evaluate, "get_chat_provider", return_value=ScriptedChatProvider()),
    ):
        code = evaluate.main(
            [
                "--dataset", str(dataset),
                "--variants", "hybrid",
                "--tag", "keyword",
                "--output", str(output),
            ]
        )

    printed = capsys.readouterr().out
    assert code == 0
    assert "1 cases, collection test_evaluate" in printed
    assert "hybrid                        1.000   1.000  1.000" in printed
    [report] = json.loads(output.read_text(encoding="utf-8"))
    assert report["name"] == "hybrid" and report["cases"][0]["case_id"] == "code"


@pytest.mark.parametrize(
    "arguments",
    [
        ["--dataset", "missing.jsonl"],
        ["--variants", "dense+magic"],
        ["--tag", "no-such-tag"],
    ],
)
def test_main_refuses_bad_arguments(arguments: list[str], dataset: Path) -> None:
    if "--dataset" not in arguments:
        arguments = ["--dataset", str(dataset), *arguments]
    with patch.object(evaluate, "get_settings", return_value=SETTINGS):
        with pytest.raises(SystemExit) as raised:
            evaluate.main(arguments)

    assert raised.value.code == 2
