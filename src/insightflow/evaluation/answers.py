"""Score the answers of the full workflow on a labeled set (opt-in and billable).

A case lists the facts that a correct answer states (`accept` holds phrasings that count as the
fact for the deterministic check) or says that the corpus cannot answer it (`expect_refusal`).
Each case runs the same workflow as `POST /query`, without a thread. The scores are:

- deterministic: fact recall by accepted phrasings, citation validity, source coverage (the
  labeled passages that a cited passage matches, as in `evaluation/retrieval.py`), workflow
  completion, attempts, tool errors, latency, and hosted-model calls and tokens;
- LLM judge (`ANSWER_JUDGE_SYSTEM_PROMPT`): facts stated correctly, unsupported claims, and
  refusal. An unreadable verdict leaves the case `unjudged`; it is never counted as a pass.

A `RunBudget` limits chat calls and tokens over the whole run. When it is used up, the current
case stops with `budget_exceeded` and the run ends with a partial report.
"""

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ConfigDict, Field, model_validator

from insightflow.agents.graph import AgentGraph
from insightflow.agents.nodes.answer import NO_CONTEXT_ANSWER
from insightflow.agents.nodes.verify import cited_passages
from insightflow.agents.observations import number_sources
from insightflow.agents.state import AgentState, new_turn_input
from insightflow.core.exceptions import BudgetExceededError, InsightFlowError
from insightflow.core.observability import (
    CallBudget,
    RunRecorder,
    TraceSink,
    error_code,
    export_run,
    recording,
)
from insightflow.evaluation.retrieval import (
    EvaluationModel,
    RelevantPassage,
    matches,
    normalize_text,
)
from insightflow.providers.instrumented import instrument_chat
from insightflow.providers.llm import ChatMessage, ChatProvider
from insightflow.providers.structured import complete_json
from insightflow.rag.models import RetrievedChunk
from insightflow.rag.retrieval import render_context_block

ANSWER_JUDGE_SYSTEM_PROMPT = (
    "You grade an answer to a question for an evaluation. You get numbered reference facts, "
    "the answer, and the numbered sources that the answer cites. "
    'Return only a JSON object: {"facts": [numbers of the reference facts that the answer '
    'states correctly], "unsupported_claims": ["..."], "refused": true or false}. '
    "A fact counts only if the answer states it; a different value is not the fact. "
    "An unsupported claim is a factual statement in the answer that the cited sources do not "
    "support; list each in a few words. "
    "Set refused to true if the answer says that it cannot find or give the information."
)


class AnswerFact(EvaluationModel):
    """A fact that a correct answer states. `accept`: phrasings that count as the fact."""

    fact: str = Field(min_length=1)
    accept: list[str] = Field(default_factory=list)


class AnswerCase(EvaluationModel):
    """One labeled question for answer evaluation."""

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    relevant: list[RelevantPassage] = Field(default_factory=list)
    facts: list[AnswerFact] = Field(default_factory=list)
    expect_refusal: bool = False
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def facts_or_refusal(self) -> "AnswerCase":
        if self.expect_refusal and (self.facts or self.relevant):
            raise ValueError("a refusal case has no facts and no relevant passages")
        if not self.expect_refusal and not self.facts:
            raise ValueError("an answerable case needs at least one fact")
        return self


class AnswerVerdict(BaseModel):
    """Answer-judge output."""

    model_config = ConfigDict(extra="ignore")

    facts: list[int] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    refused: bool = False


class AnswerResult(EvaluationModel):
    """Scores of one case. Judge fields are `None` when the case was not judged."""

    case_id: str
    tags: list[str]
    completed: bool
    error: str | None = None
    answer_chars: int = 0
    fact_recall: float | None = None
    citation_valid: bool = False
    cited: int = 0
    source_coverage: float | None = None
    refusal_correct: bool | None = None
    judged: bool = False
    judge_fact_recall: float | None = None
    unsupported_claims: int | None = None
    verification: str = ""
    attempts: int = 0
    steps: int = 0
    tool_errors: int = 0
    chat_calls: int = 0
    tokens: int = 0
    latency_ms: float = 0.0


class AnswerReport(EvaluationModel):
    """Scores of one configuration. `stopped` is set when the budget ended the run early."""

    name: str
    results: list[AnswerResult] = Field(default_factory=list)
    error: str | None = None
    stopped: str | None = None

    def mean(self, field: str) -> float | None:
        values = [
            float(value)
            for result in self.results
            if result.completed and (value := getattr(result, field)) is not None
        ]
        return fmean(values) if values else None

    def total(self, field: str) -> float:
        return float(sum(getattr(result, field) for result in self.results))


def load_answer_cases(path: Path) -> list[AnswerCase]:
    cases = [
        AnswerCase.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("answer case ids must be unique")
    return cases


@dataclass
class RunBudget:
    """Chat-call and token limits for a whole evaluation run; `None` means no limit."""

    max_chat_calls: int | None = None
    max_tokens: int | None = None
    chat_calls: int = 0
    tokens: int = 0

    def case_budget(self) -> CallBudget:
        return CallBudget(
            max_chat_calls=(
                None if self.max_chat_calls is None else self.max_chat_calls - self.chat_calls
            ),
            max_tokens=None if self.max_tokens is None else self.max_tokens - self.tokens,
        )

    def add(self, recorder: RunRecorder) -> None:
        self.chat_calls += recorder.totals("chat").calls
        self.tokens += recorder.total_tokens()


def fact_recall(case: AnswerCase, answer: str) -> float | None:
    """The share of facts with phrasings whose phrasing occurs in the answer."""
    checkable = [fact for fact in case.facts if fact.accept]
    if not checkable:
        return None
    text = normalize_text(answer)
    found = [any(normalize_text(phrase) in text for phrase in fact.accept) for fact in checkable]
    return sum(found) / len(found)


def source_coverage(case: AnswerCase, cited: Sequence[RetrievedChunk]) -> float | None:
    """The share of labeled passages that some cited passage matches."""
    if not case.relevant:
        return None
    found = [any(matches(passage, chunk) for chunk in cited) for passage in case.relevant]
    return sum(found) / len(found)


def judge_messages(
    case: AnswerCase, answer: str, cited: Sequence[tuple[int, RetrievedChunk]]
) -> list[ChatMessage]:
    facts = "\n".join(f"{number}. {fact.fact}" for number, fact in enumerate(case.facts, 1))
    sources = "\n\n".join(render_context_block(number, chunk) for number, chunk in cited)
    content = (
        f"Question: {case.question}\n\n"
        f"Reference facts:\n{facts or '(none: the documents do not answer this question)'}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Cited sources:\n{sources or '(none)'}"
    )
    return [
        {"role": "system", "content": ANSWER_JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def score_answer(
    case: AnswerCase,
    state: AgentState,
    verdict: AnswerVerdict | None,
    *,
    judged: bool,
) -> AnswerResult:
    """Score a finished workflow state; `verdict` is `None` when the judge was unreadable."""
    answer = state.get("answer", "")
    observations = state.get("observations", [])
    numbered = number_sources(observations)
    cited_numbers = cited_passages(answer)
    passages = [
        (source.number, source.passage)
        for source in numbered
        if source.passage is not None and source.number in cited_numbers
    ]
    refused_by_rule = answer.strip() == NO_CONTEXT_ANSWER
    refused = refused_by_rule or (verdict is not None and verdict.refused)
    refusal_known = refused_by_rule or verdict is not None
    verification = state.get("verification")
    result = AnswerResult(
        case_id=case.id,
        tags=list(case.tags),
        completed=True,
        answer_chars=len(answer),
        fact_recall=None if case.expect_refusal else fact_recall(case, answer),
        citation_valid=all(1 <= number <= len(numbered) for number in cited_numbers),
        cited=len(cited_numbers),
        source_coverage=source_coverage(case, [chunk for _, chunk in passages]),
        refusal_correct=(refused == case.expect_refusal) if refusal_known else None,
        judged=judged and verdict is not None,
        verification=verification.status if verification is not None else "skipped",
        attempts=state.get("attempt", 1),
        steps=len(observations),
        tool_errors=sum(1 for item in observations if item.result.status == "error"),
    )
    if verdict is None or not judged:
        return result
    valid = {number for number in verdict.facts if 1 <= number <= len(case.facts)}
    return result.model_copy(
        update={
            "judge_fact_recall": len(valid) / len(case.facts) if case.facts else None,
            "unsupported_claims": len(verdict.unsupported_claims),
        }
    )


async def run_answer_case(
    case: AnswerCase,
    graph: AgentGraph,
    config: RunnableConfig,
    judge: ChatProvider | None,
    recorder: RunRecorder,
) -> tuple[AnswerResult, AgentState | None]:
    """Run and score one case inside `recorder`. `BudgetExceededError` is raised to the caller."""
    started = time.perf_counter()
    with recording(recorder):
        try:
            state = cast(
                AgentState, await graph.ainvoke(new_turn_input(case.question), config)
            )
        except BudgetExceededError:
            raise
        except InsightFlowError as exc:
            result = AnswerResult(
                case_id=case.id, tags=list(case.tags), completed=False, error=error_code(exc)
            )
            return _with_costs(result, recorder, started), None
        verdict: AnswerVerdict | None = None
        if judge is not None:
            numbered = number_sources(state.get("observations", []))
            cited_numbers = cited_passages(state.get("answer", ""))
            cited = [
                (source.number, source.passage)
                for source in numbered
                if source.passage is not None and source.number in cited_numbers
            ]
            verdict = await complete_json(
                judge, judge_messages(case, state.get("answer", ""), cited), AnswerVerdict
            )
        result = score_answer(case, state, verdict, judged=judge is not None)
    return _with_costs(result, recorder, started), state


def _with_costs(result: AnswerResult, recorder: RunRecorder, started: float) -> AnswerResult:
    return result.model_copy(
        update={
            "chat_calls": recorder.totals("chat").calls,
            "tokens": recorder.total_tokens(),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }
    )


async def evaluate_answers(
    name: str,
    graph: AgentGraph,
    config: RunnableConfig,
    cases: Sequence[AnswerCase],
    *,
    judge: ChatProvider | None,
    budget: RunBudget,
    sink: TraceSink,
    judge_model: str = "",
) -> AnswerReport:
    """Run every case until the budget is used up; each case is one trace for the sink.

    The judge is instrumented here, so its calls count against the budget like the workflow's.
    """
    if judge is not None:
        judge = instrument_chat(judge, model=judge_model)
    results: list[AnswerResult] = []
    for case in cases:
        recorder = RunRecorder(
            f"eval-{name}-{case.id}",
            capture_content=sink.capture_content,
            budget=budget.case_budget(),
        )
        state: AgentState | None = None
        try:
            result, state = await run_answer_case(case, graph, config, judge, recorder)
        except BudgetExceededError:
            result = AnswerResult(
                case_id=case.id, tags=list(case.tags), completed=False, error="budget_exceeded"
            )
        budget.add(recorder)
        results.append(result)
        export_run(
            sink,
            recorder,
            name="evaluate_answers",
            input=case.question,
            output=state.get("answer") if state is not None else None,
            metadata={"variant": name, "case": case.id, "error": result.error},
        )
        if result.error == "budget_exceeded":
            return AnswerReport(name=name, results=results, stopped="budget")
    return AnswerReport(name=name, results=results)


def estimate_chat_calls(
    cases: int,
    *,
    max_steps: int,
    max_attempts: int,
    llm_verifier: bool,
    retrieval_chat_steps: int,
    judge: bool,
) -> tuple[int, int]:
    """Typical and worst-case chat calls for a run.

    Typical: two agent calls, one search, the answer, the verifier, and the judge. Worst case:
    every attempt uses every step, and each step is a search with the retrieval chat steps.
    """
    per_search = retrieval_chat_steps
    typical = 2 + 1 + int(llm_verifier) + per_search + int(judge)
    per_attempt = (max_steps + 1) + 1 + int(llm_verifier) + max_steps * per_search
    worst = max_attempts * per_attempt + int(judge)
    return cases * typical, cases * worst


def format_answer_reports(reports: Sequence[AnswerReport]) -> str:
    """One row per configuration: quality, then reliability, then cost."""
    header = (
        f"{'variant':<24} {'done':>5} {'facts':>6} {'judge':>6} {'unsup':>6} {'refuse':>7} "
        f"{'cite_ok':>8} {'cover':>6} {'passed':>7} {'retry':>6} {'calls':>6} {'tokens':>8} "
        f"{'ms':>7}"
    )
    lines = [header, "-" * len(header)]
    for report in reports:
        if report.error is not None:
            lines.append(f"{report.name:<24} not run: {report.error}")
            continue
        completed = [result for result in report.results if result.completed]
        done = f"{len(completed)}/{len(report.results)}"
        passed = (
            fmean(1.0 if r.verification == "passed" else 0.0 for r in completed)
            if completed
            else None
        )
        retries = fmean(r.attempts - 1 for r in completed) if completed else None
        lines.append(
            f"{report.name:<24} {done:>5} {_fmt(report.mean('fact_recall')):>6} "
            f"{_fmt(report.mean('judge_fact_recall')):>6} "
            f"{_fmt(report.mean('unsupported_claims')):>6} "
            f"{_fmt(report.mean('refusal_correct')):>7} {_fmt(report.mean('citation_valid')):>8} "
            f"{_fmt(report.mean('source_coverage')):>6} {_fmt(passed):>7} {_fmt(retries):>6} "
            f"{report.total('chat_calls'):>6.0f} {report.total('tokens'):>8.0f} "
            f"{_fmt(report.mean('latency_ms'), '.0f'):>7}"
        )
        if report.stopped:
            lines.append(f"{'':<24} stopped early: {report.stopped} limit reached")
        errors = sorted({r.error for r in report.results if r.error})
        if errors:
            lines.append(f"{'':<24} errors: {', '.join(errors)}")
    return "\n".join(lines)


def _fmt(value: float | None, spec: str = ".3f") -> str:
    return "-" if value is None else format(value, spec)


def report_json(reports: Sequence[AnswerReport]) -> list[dict[str, Any]]:
    return [report.model_dump() for report in reports]

