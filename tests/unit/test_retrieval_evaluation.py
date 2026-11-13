"""Tests for the labeled retrieval evaluation and its command-line variants."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from insightflow.evaluate import DEFAULT_DATASET, DEFAULT_VARIANTS, variant_config
from insightflow.evaluation.retrieval import (
    CaseResult,
    EvaluationCase,
    RelevantPassage,
    VariantReport,
    evaluate_retriever,
    format_reports,
    load_cases,
    matches,
    score_case,
)
from insightflow.rag.config import RetrievalConfig
from insightflow.rag.ingestion import MIME_TYPES_BY_SUFFIX
from insightflow.rag.models import ContextBudget
from insightflow.rag.retrieval import BudgetedContextAssembler
from tests.fakes import StaticRetriever, make_candidate, word_count

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def case(*relevant: RelevantPassage, tags: list[str] | None = None) -> EvaluationCase:
    return EvaluationCase(id="c1", question="Q?", relevant=list(relevant), tags=tags or [])


HANDBOOK = RelevantPassage(source="PDF/handbook.pdf", evidence=["Quiet hours start at 10 PM"])
CALENDAR = RelevantPassage(source="DOCX/calendar.docx", evidence=["August 25", "25 August"])


def test_a_passage_matches_by_source_and_normalized_evidence() -> None:
    right = make_candidate(
        "a", "Rules.\nQUIET   hours start\nat 10 pm in halls.", source="PDF/handbook.pdf"
    )
    wrong_source = make_candidate("b", "Quiet hours start at 10 PM", source="PDF/other.pdf")
    wrong_text = make_candidate("c", "Quiet hours end at 7 AM", source="PDF/handbook.pdf")

    assert matches(HANDBOOK, right)
    assert not matches(HANDBOOK, wrong_source)
    assert not matches(HANDBOOK, wrong_text)
    assert matches(RelevantPassage(source="PDF/handbook.pdf"), wrong_text)


def test_score_case_reports_hit_recall_and_reciprocal_rank() -> None:
    context = [
        make_candidate("x", "Unrelated.", source="PDF/handbook.pdf"),
        make_candidate("a", "Quiet hours start at 10 PM.", source="PDF/handbook.pdf", rank=2),
    ]

    partial = score_case(case(HANDBOOK, CALENDAR), context, context_tokens=9, latency_ms=3.0)
    miss = score_case(case(CALENDAR), context)

    assert (partial.hit, partial.recall, partial.reciprocal_rank) == (True, 0.5, 0.5)
    assert (partial.passages, partial.context_tokens, partial.latency_ms) == (2, 9, 3.0)
    assert (miss.hit, miss.recall, miss.reciprocal_rank) == (False, 0.0, 0.0)


async def test_evaluation_scores_the_assembled_context_not_all_candidates() -> None:
    candidates = [
        make_candidate("x", "Unrelated text.", source="PDF/handbook.pdf", rank=1),
        make_candidate("a", "Quiet hours start at 10 PM.", source="PDF/handbook.pdf", rank=2),
    ]
    retriever = StaticRetriever(candidates)
    assembler = BudgetedContextAssembler(word_count)

    wide = await evaluate_retriever(
        "wide", retriever, assembler, [case(HANDBOOK)], top_k=7,
        budget=ContextBudget(max_tokens=100, max_chunks=2),
    )
    narrow = await evaluate_retriever(
        "narrow", retriever, assembler, [case(HANDBOOK)], top_k=7,
        budget=ContextBudget(max_tokens=100, max_chunks=1),
    )

    assert retriever.queries[0].text == "Q?" and retriever.queries[0].top_k == 7
    assert wide.mean("hit") == 1.0 and wide.mean("reciprocal_rank") == 0.5
    assert narrow.mean("hit") == 0.0
    assert wide.cases[0].context_tokens > narrow.cases[0].context_tokens > 0


def result(case_id: str, recall: float, tags: list[str]) -> CaseResult:
    return CaseResult(
        case_id=case_id, tags=tags, hit=recall > 0, recall=recall,
        reciprocal_rank=recall, passages=2, context_tokens=100, latency_ms=10.0,
    )


def test_report_means_can_be_limited_to_a_tag_and_formatted() -> None:
    report = VariantReport(
        name="dense",
        cases=[result("a", 1.0, ["keyword"]), result("b", 0.0, ["multi_hop"])],
    )
    failed = VariantReport(name="dense+rerank", error="LITELLM_RERANK_MODEL is not set")

    table = format_reports([report, failed])

    assert report.mean("recall") == 0.5
    assert report.mean("recall", "keyword") == 1.0
    assert report.mean("recall", "absent") == 0.0
    assert "dense                         0.500   0.500  0.500" in table
    assert "dense+rerank                 not run: LITELLM_RERANK_MODEL is not set" in table
    assert "recall by tag" in table and "multi_hop" in table


def test_load_cases_reads_json_lines_and_refuses_duplicates(tmp_path: Path) -> None:
    line = json.dumps({"id": "a", "question": "Q?", "relevant": [{"source": "TXT/a.txt"}]})
    good = tmp_path / "good.jsonl"
    good.write_text(f"{line}\n\n", encoding="utf-8")
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(f"{line}\n{line}\n", encoding="utf-8")

    assert [item.id for item in load_cases(good)] == ["a"]
    with pytest.raises(ValueError, match="unique"):
        load_cases(duplicate)


def test_variants_add_options_to_plain_dense_and_keep_configured_sizes() -> None:
    configured = RetrievalConfig(
        mode="hybrid", corrective=True, candidate_top_k=20, rerank_top_n=4, neighbors_per_hit=2
    )

    dense = variant_config("dense", configured)
    combined = variant_config("hybrid+rerank+neighbors", configured)

    assert variant_config("configured", configured) is configured
    assert dense == RetrievalConfig(candidate_top_k=20, rerank_top_n=4)
    assert (combined.mode, combined.rerank, combined.corrective) == ("hybrid", True, False)
    assert combined.neighbors_per_hit == 2
    assert variant_config("dense+neighbors", RetrievalConfig()).neighbors_per_hit == 1
    assert variant_config("dense+decompose", configured).query_expansion == "decompose"
    for name in DEFAULT_VARIANTS:
        variant_config(name, configured)
    with pytest.raises(ValueError, match="unknown variant option"):
        variant_config("dense+magic", configured)


def test_the_repository_evaluation_set_is_valid_and_points_at_corpus_files() -> None:
    cases = load_cases(REPOSITORY_ROOT / DEFAULT_DATASET)

    assert len(cases) >= 30
    for item in cases:
        assert item.tags, item.id
        for passage in item.relevant:
            path = REPOSITORY_ROOT / "corpus" / passage.source
            assert path.is_file(), f"{item.id}: {passage.source}"
            assert path.suffix.lower() in MIME_TYPES_BY_SUFFIX
            assert passage.evidence, item.id
