"""Command-line entry point that compares retrieval configurations on a labeled set.

    PYTHONPATH=src python -m insightflow.evaluate
    PYTHONPATH=src python -m insightflow.evaluate --variants dense,hybrid,hybrid+rerank

It reads the indexed collection and makes billable requests: one query embedding per case and
variant, and chat or rerank calls for the variants that use them. It writes nothing to Qdrant.
"""

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from insightflow.api.dependencies import (
    get_chat_provider,
    get_embedding_provider,
    get_rerank_provider,
    retrieval_config_from_settings,
)
from insightflow.core.config import Settings, get_settings
from insightflow.core.exceptions import InsightFlowError
from insightflow.evaluation.retrieval import (
    EvaluationCase,
    VariantReport,
    evaluate_retriever,
    format_reports,
    load_cases,
)
from insightflow.rag.composition import build_retriever
from insightflow.rag.config import RetrievalConfig
from insightflow.rag.models import ContextBudget
from insightflow.rag.retrieval import BudgetedContextAssembler
from insightflow.storage.qdrant import QdrantStore

DEFAULT_DATASET = Path("evaluation/retrieval_cases.jsonl")
CONFIGURED = "configured"
OPTIONS: dict[str, dict[str, Any]] = {
    "dense": {"mode": "dense"},
    "hybrid": {"mode": "hybrid"},
    "multi_query": {"query_expansion": "multi_query"},
    "decompose": {"query_expansion": "decompose"},
    "rerank": {"rerank": True},
    "corrective": {"corrective": True},
    "neighbors": {"neighbors_per_hit": 1},
    "compress": {"compress_context": True},
}
DEFAULT_VARIANTS = (
    "dense",
    "hybrid",
    "dense+multi_query",
    "dense+decompose",
    "dense+rerank",
    "dense+corrective",
    "dense+neighbors",
    "dense+compress",
    "hybrid+rerank",
)


def variant_config(name: str, configured: RetrievalConfig) -> RetrievalConfig:
    """Build a variant: `configured` is the `.env` setup; `a+b` adds options to plain dense.

    Sizes (candidates, context budget, rerank top-n) always come from the settings, so the
    variants differ only in technique.
    """
    if name == CONFIGURED:
        return configured
    updates: dict[str, Any] = {
        "mode": "dense",
        "query_expansion": "off",
        "rerank": False,
        "corrective": False,
        "neighbors_per_hit": 0,
        "compress_context": False,
    }
    for option in name.split("+"):
        if option not in OPTIONS:
            raise ValueError(f"unknown variant option {option!r}")
        updates.update(OPTIONS[option])
    if updates["neighbors_per_hit"] and configured.neighbors_per_hit:
        updates["neighbors_per_hit"] = configured.neighbors_per_hit
    return RetrievalConfig.model_validate({**configured.model_dump(), **updates})


async def run(
    cases: Sequence[EvaluationCase], variants: Sequence[str], settings: Settings
) -> list[VariantReport]:
    """Evaluate each variant against the configured collection."""
    configured = retrieval_config_from_settings(settings)
    embedding_provider = get_embedding_provider(settings)
    assembler = BudgetedContextAssembler(embedding_provider.count_tokens)
    budget = ContextBudget(
        max_tokens=configured.context_token_budget, max_chunks=configured.generation_max_chunks
    )
    store = QdrantStore(settings)
    reports: list[VariantReport] = []
    try:
        for name in variants:
            config = variant_config(name, configured)
            rerank_provider = get_rerank_provider(settings)
            if config.rerank and rerank_provider is None:
                reports.append(VariantReport(name=name, error="LITELLM_RERANK_MODEL is not set"))
                continue
            retriever = build_retriever(
                config,
                embedding_provider=embedding_provider,
                searcher=store,
                reader=store,
                chat_provider=get_chat_provider(settings),
                rerank_provider=rerank_provider,
            )
            try:
                report = await evaluate_retriever(
                    name,
                    retriever,
                    assembler,
                    cases,
                    top_k=config.candidate_top_k,
                    budget=budget,
                )
            except InsightFlowError as exc:
                report = VariantReport(name=name, error=f"{type(exc).__name__}: {exc}")
            reports.append(report)
            print(f"finished {name}", file=sys.stderr)
    finally:
        await store.close()
    return reports


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare retrieval configurations on a labeled evaluation set (billable)."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help=(
            f"comma-separated; '{CONFIGURED}' is the .env setup, and options joined with '+' "
            f"are added to plain dense retrieval: {', '.join(OPTIONS)}"
        ),
    )
    parser.add_argument("--tag", help="evaluate only the cases with this tag")
    parser.add_argument("--output", type=Path, help="also write the per-case scores as JSON")
    args = parser.parse_args(argv)

    if not args.dataset.is_file():
        parser.error(f"dataset does not exist: {args.dataset}")
    cases = load_cases(args.dataset)
    if args.tag:
        cases = [case for case in cases if args.tag in case.tags]
    if not cases:
        parser.error("no evaluation cases selected")
    variants = [name.strip() for name in args.variants.split(",") if name.strip()]
    settings = get_settings()
    try:
        configured = retrieval_config_from_settings(settings)
        for name in variants:
            variant_config(name, configured)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        reports = asyncio.run(run(cases, variants, settings))
    except InsightFlowError as exc:
        print(f"Evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"{len(cases)} cases, collection {settings.qdrant_collection}")
    print(format_reports(reports))
    if args.output is not None:
        args.output.write_text(
            json.dumps([report.model_dump() for report in reports], indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
