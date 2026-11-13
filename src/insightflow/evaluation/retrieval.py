"""Score what a retrieval configuration puts into the model's context.

A case names the passages that hold the answer by source document and by evidence text, not by
chunk identifier, so the labels stay valid when chunking or parser versions change. The scores
are computed on the assembled context (after the chunk and token budget), because that is what
the answer model reads.
"""

import json
import re
import time
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean
from unicodedata import normalize

from pydantic import BaseModel, ConfigDict, Field

from insightflow.rag.models import ContextBudget, RetrievalQuery, RetrievedChunk
from insightflow.rag.protocols import ContextAssembler, Retriever

_WHITESPACE_RE = re.compile(r"\s+")


class EvaluationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RelevantPassage(EvaluationModel):
    """A labeled answer location: a source and text that a relevant passage contains.

    A passage matches when it comes from `source` and contains at least one `evidence` text
    (case and whitespace are ignored). Without evidence, every passage of the source matches.
    """

    source: str = Field(min_length=1)
    evidence: list[str] = Field(default_factory=list)


class EvaluationCase(EvaluationModel):
    """One labeled question. `relevant` lists every fact location that the answer needs."""

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    relevant: list[RelevantPassage] = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)


class CaseResult(EvaluationModel):
    """Scores of one case for one configuration."""

    case_id: str
    tags: list[str]
    hit: bool
    recall: float
    reciprocal_rank: float
    passages: int
    context_tokens: int
    latency_ms: float


class VariantReport(EvaluationModel):
    """Scores of one configuration over the evaluation set."""

    name: str
    cases: list[CaseResult] = Field(default_factory=list)
    error: str | None = None

    def mean(self, field: str, tag: str | None = None) -> float:
        values = [
            float(getattr(case, field))
            for case in self.cases
            if tag is None or tag in case.tags
        ]
        return fmean(values) if values else 0.0


def load_cases(path: Path) -> list[EvaluationCase]:
    """Read a JSON Lines evaluation set; blank lines are ignored and ids must be unique."""
    cases = [
        EvaluationCase.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("evaluation case ids must be unique")
    return cases


def normalize_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", normalize("NFKC", text)).strip().casefold()


def matches(passage: RelevantPassage, candidate: RetrievedChunk) -> bool:
    """True when the candidate is from the labeled source and holds its evidence."""
    if candidate.source_identifier != passage.source:
        return False
    if not passage.evidence:
        return True
    content = normalize_text(candidate.chunk.content)
    return any(normalize_text(evidence) in content for evidence in passage.evidence)


def score_case(
    case: EvaluationCase,
    context: Sequence[RetrievedChunk],
    *,
    context_tokens: int = 0,
    latency_ms: float = 0.0,
) -> CaseResult:
    """Score the passages of an assembled context, in their context order.

    `hit`: some labeled passage is in the context. `recall`: the share of labeled passages
    that are in the context (below 1 means that a multi-fact question lacks a fact).
    `reciprocal_rank`: 1 / position of the first relevant passage.
    """
    found = [
        any(matches(passage, candidate) for candidate in context) for passage in case.relevant
    ]
    first = next(
        (
            position
            for position, candidate in enumerate(context, start=1)
            if any(matches(passage, candidate) for passage in case.relevant)
        ),
        None,
    )
    return CaseResult(
        case_id=case.id,
        tags=list(case.tags),
        hit=any(found),
        recall=sum(found) / len(found),
        reciprocal_rank=1.0 / first if first else 0.0,
        passages=len(context),
        context_tokens=context_tokens,
        latency_ms=latency_ms,
    )


async def evaluate_retriever(
    name: str,
    retriever: Retriever,
    assembler: ContextAssembler,
    cases: Sequence[EvaluationCase],
    *,
    top_k: int,
    budget: ContextBudget,
) -> VariantReport:
    """Run every case through retrieval and context assembly, as `search_documents` does."""
    results: list[CaseResult] = []
    for case in cases:
        started = time.perf_counter()
        candidates = await retriever.retrieve(RetrievalQuery(text=case.question, top_k=top_k))
        latency_ms = (time.perf_counter() - started) * 1000
        context = assembler.assemble(case.question, candidates, budget)
        results.append(
            score_case(
                case,
                context.chunks,
                context_tokens=context.token_count,
                latency_ms=latency_ms,
            )
        )
    return VariantReport(name=name, cases=results)


def format_reports(reports: Sequence[VariantReport]) -> str:
    """Render one row per configuration, then hit rates per tag."""
    header = (
        f"{'variant':<28} {'hit':>6} {'recall':>7} {'mrr':>6} "
        f"{'passages':>9} {'tokens':>7} {'ms':>7}"
    )
    lines = [header, "-" * len(header)]
    for report in reports:
        if report.error is not None:
            lines.append(f"{report.name:<28} not run: {report.error}")
            continue
        lines.append(
            f"{report.name:<28} {report.mean('hit'):>6.3f} {report.mean('recall'):>7.3f} "
            f"{report.mean('reciprocal_rank'):>6.3f} {report.mean('passages'):>9.1f} "
            f"{report.mean('context_tokens'):>7.0f} {report.mean('latency_ms'):>7.0f}"
        )
    completed = [report for report in reports if report.error is None]
    tags = sorted({tag for report in completed for case in report.cases for tag in case.tags})
    if tags and completed:
        lines.append("")
        lines.append("recall by tag")
        lines.append(f"{'variant':<28} " + " ".join(f"{tag:>12}" for tag in tags))
        for report in completed:
            lines.append(
                f"{report.name:<28} "
                + " ".join(f"{report.mean('recall', tag):>12.3f}" for tag in tags)
            )
    return "\n".join(lines)
