"""Command-line entry point that ingests a corpus directory into Qdrant."""

import argparse
import asyncio
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from insightflow.core.config import Settings, get_settings
from insightflow.core.exceptions import InsightFlowError, ProviderConfigurationError
from insightflow.providers.embeddings import LiteLLMEmbeddingProvider
from insightflow.rag.config import ChunkingConfig, EmbeddingConfig, IngestionConfig
from insightflow.rag.ingestion import IngestionPipeline, create_chunker, create_default_parsers
from insightflow.rag.models import IngestionResult
from insightflow.storage.qdrant import QdrantStore


def chunking_config_from_settings(settings: Settings) -> ChunkingConfig:
    return ChunkingConfig(
        strategy=settings.rag_chunking_strategy,
        target_tokens=settings.rag_chunk_target_tokens,
        max_tokens=settings.rag_chunk_max_tokens,
        min_tokens=settings.rag_chunk_min_tokens,
        overlap_tokens=settings.rag_chunk_overlap_tokens,
        hard_threshold=settings.rag_max_min_hard_threshold,
        similarity_coefficient=settings.rag_max_min_similarity_coefficient,
        initialization_constant=settings.rag_max_min_initialization_constant,
    )


def embedding_config_from_settings(settings: Settings) -> EmbeddingConfig:
    """Require the embedding settings that define the collection's vector shape."""
    dimensions = settings.embedding_dimensions
    max_input_tokens = settings.embedding_max_input_tokens
    missing = [
        name
        for name, value in (
            ("EMBEDDING_PROVIDER", settings.embedding_provider),
            ("LITELLM_EMBEDDING_MODEL", settings.litellm_embedding_model),
            ("EMBEDDING_DIMENSIONS", dimensions),
            ("EMBEDDING_MAX_INPUT_TOKENS", max_input_tokens),
        )
        if not value
    ]
    if missing or dimensions is None or max_input_tokens is None:
        raise ProviderConfigurationError(f"Missing settings: {', '.join(missing)}")
    return EmbeddingConfig(
        provider=settings.embedding_provider,
        model=settings.litellm_embedding_model,
        dimensions=dimensions,
        max_input_tokens=max_input_tokens,
        distance=settings.qdrant_distance,
    )


def format_result(result: IngestionResult) -> str:
    codes = ",".join(warning.code for warning in result.warnings)
    line = f"{result.status:<9} {result.chunk_count:>5}  {result.source_identifier}"
    return f"{line}  [{codes}]" if codes else line


async def run(corpus_dir: Path, settings: Settings, *, force: bool) -> list[IngestionResult]:
    """Compose the configured adapters and ingest one corpus directory."""
    chunking_config = chunking_config_from_settings(settings)
    embedding_config = embedding_config_from_settings(settings)
    embedding_provider = LiteLLMEmbeddingProvider(
        model=settings.litellm_embedding_model,
        api_key=settings.litellm_api_key,
        api_base=settings.litellm_api_base,
    )
    store = QdrantStore(settings)
    try:
        pipeline = IngestionPipeline(
            parsers=create_default_parsers(),
            chunker=create_chunker(
                chunking_config,
                embedding_provider,
                embedding_batch_size=settings.rag_embedding_batch_size,
            ),
            chunking_config=chunking_config,
            embedding_provider=embedding_provider,
            embedding_config=embedding_config,
            store=store,
            ingestion_config=IngestionConfig(
                embedding_batch_size=settings.rag_embedding_batch_size,
                upload_batch_size=settings.rag_upload_batch_size,
                force=force,
            ),
        )
        return await pipeline.ingest_corpus(corpus_dir)
    finally:
        await store.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest a corpus directory into Qdrant.")
    parser.add_argument(
        "corpus_dir",
        nargs="?",
        type=Path,
        help="corpus directory (default: RAG_CORPUS_DIR)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-ingest documents even when their indexed version is unchanged",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    corpus_dir: Path | None = args.corpus_dir or settings.rag_corpus_dir
    if corpus_dir is None:
        parser.error("provide CORPUS_DIR or set RAG_CORPUS_DIR")
    if not corpus_dir.is_dir():
        parser.error(f"corpus directory does not exist: {corpus_dir}")

    try:
        results = asyncio.run(run(corpus_dir, settings, force=args.force))
    except InsightFlowError as exc:
        print(f"Ingestion failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    for result in results:
        print(format_result(result))
    totals = Counter(result.status for result in results)
    summary = ", ".join(f"{totals[status]} {status}" for status in sorted(totals))
    print(f"{len(results)} documents: {summary or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
