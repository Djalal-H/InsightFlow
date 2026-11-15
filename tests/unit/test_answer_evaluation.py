"""Tests for answer-quality scoring, the run budget, and the evaluate_answers command."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from insightflow import evaluate_answers as cli
from insightflow.agents.models import Observation, VerificationResult
from insightflow.agents.nodes.answer import NO_CONTEXT_ANSWER
from insightflow.agents.state import AgentState
from insightflow.api.workflow import build_workflow, resolve_thread
from insightflow.core.config import Settings
from insightflow.core.exceptions import ProviderTimeoutError
from insightflow.core.observability import NullTraceSink, RunTrace
from insightflow.evaluation.answers import (
    AnswerCase,
    AnswerReport,
    AnswerResult,
    AnswerVerdict,
    RunBudget,
    estimate_chat_calls,
    evaluate_answers,
    fact_recall,
    format_answer_reports,
    load_answer_cases,
    score_answer,
)
from insightflow.rag.retrieval import BudgetedContextAssembler
from insightflow.tools.models import ToolResult
from tests.fakes import ANSWER, ScriptedChatProvider, StaticRetriever, make_candidate, search

REPOSITORY = Path(__file__).parents[2]
SETTINGS = Settings(
    _env_file=None,
    litellm_chat_model="test/chat",
    litellm_rerank_model="",  # LiteLLM loads .env into the environment; be explicit
    langfuse_enabled=False,
    agent_verifier_llm_enabled=False,
    agent_max_attempts=1,
)
FEE = make_candidate(content="The library late fee is $2 per day.", source="PDF/handbook.pdf")


def case(**fields: Any) -> AnswerCase:
    values: dict[str, Any] = {
        "id": "fee",
        "question": "What is the late fee?",
        "relevant": [{"source": "PDF/handbook.pdf", "evidence": ["late fee"]}],
        "facts": [{"fact": "The fee is $2 per day", "accept": ["$2", "2 dollars"]}],
    }
    values.update(fields)
    return AnswerCase.model_validate(values)


def refusal_case() -> AnswerCase:
    return AnswerCase(id="none", question="Q?", expect_refusal=True, tags=["refusal"])


def search_state(answer: str) -> AgentState:
    result = ToolResult(
        call_id="c1",
        server="insightflow",
        tool="search_documents",
        arguments={"query": "late fee"},
        status="ok",
        output="",
        started_at="2026-11-14T00:00:00Z",  # type: ignore[arg-type]
        duration_ms=1.0,
    )
    return {
        "answer": answer,
        "observations": [Observation(attempt=1, result=result, passages=[FEE])],
        "verification": VerificationResult(status="passed", method="deterministic"),
        "attempt": 1,
    }


def test_the_repository_answer_set_is_valid() -> None:
    cases = load_answer_cases(REPOSITORY / "evaluation" / "answer_cases.jsonl")
    assert len(cases) >= 20
    assert any(item.expect_refusal for item in cases)
    for item in cases:
        for passage in item.relevant:
            assert (REPOSITORY / "corpus" / passage.source).is_file(), passage.source


def test_cases_need_facts_or_a_refusal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one fact"):
        case(facts=[])
    with pytest.raises(ValueError, match="refusal case"):
        case(expect_refusal=True)
    duplicate = tmp_path / "cases.jsonl"
    line = refusal_case().model_dump_json()
    duplicate.write_text(f"{line}\n\n{line}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_answer_cases(duplicate)


def test_fact_recall_matches_accepted_phrasings_ignoring_case_and_space() -> None:
    item = case(
        facts=[
            {"fact": "fee", "accept": ["$2  PER day"]},
            {"fact": "unit", "accept": ["per week"]},
            {"fact": "judge only"},
        ]
    )
    assert fact_recall(item, "It is $2 per DAY [1].") == 0.5
    assert fact_recall(case(facts=[{"fact": "judge only"}]), "x") is None


def test_a_cited_grounded_answer_is_scored_with_the_verdict() -> None:
    verdict = AnswerVerdict(facts=[1, 7], unsupported_claims=["x"])
    result = score_answer(case(), search_state("It is $2 per day [1]."), verdict, judged=True)

    assert result.completed and result.judged
    assert (result.fact_recall, result.judge_fact_recall, result.unsupported_claims) == (
        1.0,
        1.0,
        1,
    )
    assert (result.citation_valid, result.cited, result.source_coverage) == (True, 1, 1.0)
    assert (result.refusal_correct, result.verification, result.steps) == (True, "passed", 1)


def test_invalid_citations_and_uncited_sources_are_scored() -> None:
    result = score_answer(case(), search_state("It is $2 [3]."), None, judged=True)
    assert (result.citation_valid, result.source_coverage) == (False, 0.0)
    assert not result.judged  # unreadable verdict
    assert (result.judge_fact_recall, result.unsupported_claims, result.refusal_correct) == (
        None,
        None,
        None,
    )


def test_refusals_are_recognized_by_rule_or_by_the_judge() -> None:
    no_context: AgentState = {"answer": NO_CONTEXT_ANSWER, "observations": []}
    assert score_answer(refusal_case(), no_context, None, judged=False).refusal_correct is True
    judged = score_answer(
        refusal_case(), {"answer": "I cannot say."}, AnswerVerdict(refused=True), judged=True
    )
    assert judged.refusal_correct is True
    assert judged.fact_recall is None and judged.source_coverage is None
    wrong = score_answer(case(), {"answer": "No idea."}, AnswerVerdict(refused=True), judged=True)
    assert wrong.refusal_correct is False


def graph_for(chat: ScriptedChatProvider, retriever: StaticRetriever) -> Any:
    return build_workflow(
        SETTINGS,
        chat,
        retriever,
        BudgetedContextAssembler(lambda text: len(text.split())),
        None,
    )


class Sink(NullTraceSink):
    def __init__(self) -> None:
        self.traces: list[RunTrace] = []

    def export(self, trace: RunTrace) -> None:
        self.traces.append(trace)


async def test_evaluation_runs_the_workflow_and_judges_each_case() -> None:
    chat = ScriptedChatProvider(
        agent=[search("late fee"), ANSWER, search("late fee"), ANSWER],
        answer="It is $2 per day [1].",
        evaluate=['{"facts": [1], "unsupported_claims": [], "refused": false}', "unreadable"],
    )
    _, config = resolve_thread(None, None, SETTINGS)
    sink = Sink()
    budget = RunBudget()

    report = await evaluate_answers(
        "dense",
        graph_for(chat, StaticRetriever([FEE])),
        config,
        [case(), case(id="fee-2")],
        judge=chat,
        budget=budget,
        sink=sink,
    )

    first, second = report.results
    assert first.judged and first.judge_fact_recall == 1.0 and first.fact_recall == 1.0
    assert not second.judged and second.judge_fact_recall is None
    assert first.chat_calls == 4  # agent, agent, answer, judge
    assert budget.chat_calls == 8
    assert report.stopped is None
    assert [trace.metadata["case"] for trace in sink.traces] == ["fee", "fee-2"]
    assert all(trace.input is None for trace in sink.traces)


async def test_the_budget_stops_the_run_with_a_partial_report() -> None:
    chat = ScriptedChatProvider()
    _, config = resolve_thread(None, None, SETTINGS)
    budget = RunBudget(max_chat_calls=6)

    report = await evaluate_answers(
        "dense",
        graph_for(chat, StaticRetriever([FEE])),
        config,
        [case(id="a"), case(id="b"), case(id="c")],
        judge=chat,
        budget=budget,
        sink=NullTraceSink(),
    )

    assert [result.case_id for result in report.results] == ["a", "b"]
    assert report.results[0].completed
    assert report.results[1].error == "budget_exceeded"
    assert report.stopped == "budget"
    assert budget.chat_calls == 6


async def test_a_provider_failure_marks_the_case_and_the_run_continues() -> None:
    chat = ScriptedChatProvider(answer=[ProviderTimeoutError(), "It is $2 [1]."])
    _, config = resolve_thread(None, None, SETTINGS)

    report = await evaluate_answers(
        "dense",
        graph_for(chat, StaticRetriever([FEE])),
        config,
        [case(id="a"), case(id="b")],
        judge=None,
        budget=RunBudget(),
        sink=NullTraceSink(),
    )

    failed, done = report.results
    assert (failed.completed, failed.error) == (False, "provider_timeout")
    assert done.completed and not done.judged


def test_estimates_and_the_run_budget() -> None:
    assert estimate_chat_calls(
        10, max_steps=4, max_attempts=2, llm_verifier=True, retrieval_chat_steps=1, judge=True
    ) == (60, 10 * (2 * (5 + 1 + 1 + 4) + 1))
    budget = RunBudget(max_chat_calls=10, max_tokens=100, chat_calls=4, tokens=30)
    limits = budget.case_budget()
    assert (limits.max_chat_calls, limits.max_tokens) == (6, 70)
    assert RunBudget().case_budget().max_chat_calls is None


def test_reports_are_formatted_with_gaps_and_stops() -> None:
    done = AnswerResult(
        case_id="a",
        tags=[],
        completed=True,
        fact_recall=1.0,
        citation_valid=True,
        verification="passed",
        attempts=2,
        chat_calls=4,
        tokens=120,
    )
    failed = AnswerResult(case_id="b", tags=[], completed=False, error="budget_exceeded")
    text = format_answer_reports(
        [
            AnswerReport(name="dense", results=[done, failed], stopped="budget"),
            AnswerReport(name="dense+rerank", error="LITELLM_RERANK_MODEL is not set"),
            AnswerReport(name="empty"),
        ]
    )
    row = next(line for line in text.splitlines() if line.startswith("dense "))
    assert row.split()[:4] == ["dense", "1/2", "1.000", "-"]
    assert "stopped early: budget limit reached" in text
    assert "errors: budget_exceeded" in text
    assert "dense+rerank             not run: LITELLM_RERANK_MODEL is not set" in text
    assert AnswerReport(name="x").mean("fact_recall") is None


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "answers.jsonl"
    lines = [case().model_dump_json(), refusal_case().model_dump_json()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_dry_run_prints_the_estimate_and_makes_no_request(
    dataset: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with (
        patch.object(cli, "get_settings", return_value=SETTINGS),
        patch.object(cli, "run") as run,
    ):
        code = cli.main(
            ["--dataset", str(dataset), "--variants", "dense,dense+rerank", "--dry-run"]
        )

    assert code == 0
    run.assert_not_called()
    printed = capsys.readouterr().out
    assert "2 cases x 2 variants; estimated chat calls: about 16, at most 28" in printed
    assert "model test/chat" in printed


def test_main_runs_prints_and_writes_json(
    dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "answers.json"
    chat = ScriptedChatProvider(
        agent=[search("late fee"), ANSWER, ANSWER], answer="It is $2 per day [1]."
    )

    class Store:
        def __init__(self, settings: Settings) -> None:
            del settings

        async def close(self) -> None:
            return None

    with (
        patch.object(cli, "get_settings", return_value=SETTINGS),
        patch.object(cli, "QdrantStore", Store),
        patch.object(cli, "get_chat_provider", return_value=chat),
        patch.object(cli, "get_embedding_provider", return_value=FakeEmbedding()),
        patch.object(cli, "build_retriever", return_value=StaticRetriever([FEE])),
        patch.object(cli, "configure_logging"),
    ):
        code = cli.main(
            ["--dataset", str(dataset), "--output", str(output), "--max-chat-calls", "50"]
        )

    printed = capsys.readouterr().out
    assert code == 0, printed
    assert "2 cases, collection" in printed
    assert "configured" in printed and "total:" in printed
    [report] = json.loads(output.read_text(encoding="utf-8"))
    assert [result["case_id"] for result in report["results"]] == ["fee", "none"]


class FakeEmbedding:
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [1.0]

    def count_tokens(self, text: str) -> int:
        return len(text.split())


def test_rerank_variants_need_a_rerank_model(dataset: Path) -> None:
    import asyncio

    class Store:
        def __init__(self, settings: Settings) -> None:
            del settings

        async def close(self) -> None:
            return None

    with (
        patch.object(cli, "QdrantStore", Store),
        patch.object(cli, "get_chat_provider", return_value=ScriptedChatProvider()),
        patch.object(cli, "get_embedding_provider", return_value=FakeEmbedding()),
    ):
        [report] = asyncio.run(
            cli.run(
                load_answer_cases(dataset),
                ["dense+rerank"],
                SETTINGS,
                budget=RunBudget(),
                sink=NullTraceSink(),
            )
        )
    assert report.error == "LITELLM_RERANK_MODEL is not set"


@pytest.mark.parametrize(
    "arguments",
    [
        ["--dataset", "missing.jsonl"],
        ["--variants", "dense+magic"],
        ["--tag", "no-such-tag"],
        ["--max-cases", "0"],
    ],
)
def test_main_refuses_bad_arguments(arguments: list[str], dataset: Path) -> None:
    if "--dataset" not in arguments:
        arguments = ["--dataset", str(dataset), *arguments]
    with patch.object(cli, "get_settings", return_value=SETTINGS):
        with pytest.raises(SystemExit) as raised:
            cli.main(arguments)
    assert raised.value.code == 2
