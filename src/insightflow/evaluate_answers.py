"""Command-line entry point that scores workflow answers on a labeled set (billable).

    PYTHONPATH=src python -m insightflow.evaluate_answers --dry-run
    PYTHONPATH=src python -m insightflow.evaluate_answers --variants configured,dense+rerank

Every case runs the full agent workflow (as `POST /query`, without a thread or tools unless
`--with-tools`) against the configured collection, then one judge call. The run stops when
`--max-chat-calls` or `--max-tokens` is reached and prints a partial report. `--dry-run`
prints the case count and the estimated calls and makes no request. With `LANGFUSE_ENABLED`,
each case is sent as one trace.
"""

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from insightflow.api.dependencies import (
    create_tool_set,
    create_trace_sink,
    get_chat_provider,
    get_embedding_provider,
    get_rerank_provider,
    retrieval_config_from_settings,
)
from insightflow.api.workflow import build_workflow, resolve_thread
from insightflow.core.config import Settings, get_settings
from insightflow.core.exceptions import InsightFlowError
from insightflow.core.logging import configure_logging
from insightflow.core.observability import TraceSink
from insightflow.evaluate import CONFIGURED, OPTIONS, variant_config
from insightflow.evaluation.answers import (
    AnswerCase,
    AnswerReport,
    RunBudget,
    estimate_chat_calls,
    evaluate_answers,
    format_answer_reports,
    load_answer_cases,
    report_json,
)
from insightflow.providers.instrumented import (
    instrument_chat,
    instrument_embedding,
    instrument_rerank,
)
from insightflow.rag.composition import build_retriever
from insightflow.rag.config import RetrievalConfig
from insightflow.rag.retrieval import BudgetedContextAssembler
from insightflow.storage.qdrant import QdrantStore
from insightflow.tools.registry import ToolSet

DEFAULT_DATASET = Path("evaluation/answer_cases.jsonl")
DEFAULT_MAX_CHAT_CALLS = 200
DEFAULT_MAX_TOKENS = 400_000


def retrieval_chat_steps(config: RetrievalConfig) -> int:
    """Chat calls that the retrieval steps make per search."""
    return (
        int(config.query_expansion != "off")
        + int(config.corrective)
        + int(config.compress_context)
    )


async def run(
    cases: Sequence[AnswerCase],
    variants: Sequence[str],
    settings: Settings,
    *,
    budget: RunBudget,
    judge: bool = True,
    with_tools: bool = False,
    sink: TraceSink,
) -> list[AnswerReport]:
    """Evaluate each variant in turn; all variants share the budget."""
    configured = retrieval_config_from_settings(settings)
    embedding_provider = instrument_embedding(
        get_embedding_provider(settings), model=settings.litellm_embedding_model
    )
    chat_provider = instrument_chat(
        get_chat_provider(settings), model=settings.litellm_chat_model
    )
    assembler = BudgetedContextAssembler(embedding_provider.count_tokens)
    tools = create_tool_set(settings) if with_tools else ToolSet()
    _, run_config = resolve_thread(None, None, settings)
    store = QdrantStore(settings)
    reports: list[AnswerReport] = []
    try:
        for name in variants:
            config = variant_config(name, configured)
            rerank_provider = get_rerank_provider(settings)
            if config.rerank and rerank_provider is None:
                reports.append(AnswerReport(name=name, error="LITELLM_RERANK_MODEL is not set"))
                continue
            retriever = build_retriever(
                config,
                embedding_provider=embedding_provider,
                searcher=store,
                reader=store,
                chat_provider=chat_provider,
                rerank_provider=(
                    instrument_rerank(rerank_provider, model=settings.litellm_rerank_model)
                    if rerank_provider is not None
                    else None
                ),
            )
            graph = build_workflow(settings, chat_provider, retriever, assembler, None, tools=tools)
            report = await evaluate_answers(
                name,
                graph,
                run_config,
                cases,
                judge=chat_provider if judge else None,
                budget=budget,
                sink=sink,
                judge_model=settings.litellm_chat_model,
            )
            reports.append(report)
            print(
                f"finished {name}: {len(report.results)} cases, chat calls {budget.chat_calls}, "
                f"tokens {budget.tokens}",
                file=sys.stderr,
            )
            if report.stopped:
                break
    finally:
        await store.close()
        sink.flush()
    return reports


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score workflow answers on a labeled set (billable; see --dry-run)."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--variants",
        default=CONFIGURED,
        help=(
            f"comma-separated; '{CONFIGURED}' is the .env setup, and options joined with '+' "
            f"are added to plain dense retrieval: {', '.join(OPTIONS)}"
        ),
    )
    parser.add_argument("--tag", help="evaluate only the cases with this tag")
    parser.add_argument("--max-cases", type=int, help="use only the first N cases")
    parser.add_argument("--max-chat-calls", type=int, default=DEFAULT_MAX_CHAT_CALLS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--no-judge", action="store_true", help="deterministic scores only")
    parser.add_argument(
        "--with-tools", action="store_true", help="offer the configured tool servers"
    )
    parser.add_argument("--dry-run", action="store_true", help="print the estimate only")
    parser.add_argument("--output", type=Path, help="also write the per-case scores as JSON")
    args = parser.parse_args(argv)

    if not args.dataset.is_file():
        parser.error(f"dataset does not exist: {args.dataset}")
    for option in ("max_cases", "max_chat_calls", "max_tokens"):
        value = getattr(args, option)
        if value is not None and value < 1:
            parser.error(f"--{option.replace('_', '-')} must be at least 1")
    cases = load_answer_cases(args.dataset)
    if args.tag:
        cases = [case for case in cases if args.tag in case.tags]
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    if not cases:
        parser.error("no evaluation cases selected")
    variants = [name.strip() for name in args.variants.split(",") if name.strip()]
    settings = get_settings()
    configured = retrieval_config_from_settings(settings)
    try:
        configs = [variant_config(name, configured) for name in variants]
    except ValueError as exc:
        parser.error(str(exc))

    typical = worst = 0
    for config in configs:
        variant_typical, variant_worst = estimate_chat_calls(
            len(cases),
            max_steps=settings.agent_max_steps,
            max_attempts=settings.agent_max_attempts,
            llm_verifier=settings.agent_verifier_llm_enabled,
            retrieval_chat_steps=retrieval_chat_steps(config),
            judge=not args.no_judge,
        )
        typical, worst = typical + variant_typical, worst + variant_worst
    print(
        f"{len(cases)} cases x {len(variants)} variants; estimated chat calls: about {typical}, "
        f"at most {worst}; limits: {args.max_chat_calls} chat calls, {args.max_tokens} tokens; "
        f"model {settings.litellm_chat_model or '(not set)'}"
    )
    if args.dry_run:
        return 0

    configure_logging(
        settings.log_level, log_format=settings.log_format, secrets=settings.secret_values()
    )
    budget = RunBudget(max_chat_calls=args.max_chat_calls, max_tokens=args.max_tokens)
    try:
        reports = asyncio.run(
            run(
                cases,
                variants,
                settings,
                budget=budget,
                judge=not args.no_judge,
                with_tools=args.with_tools,
                sink=create_trace_sink(settings),
            )
        )
    except InsightFlowError as exc:
        print(f"Evaluation failed: {type(exc).__name__}", file=sys.stderr)
        return 1

    print(f"{len(cases)} cases, collection {settings.qdrant_collection}")
    print(format_answer_reports(reports))
    print(f"total: {budget.chat_calls} chat calls, {budget.tokens} tokens")
    if args.output is not None:
        args.output.write_text(json.dumps(report_json(reports), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
