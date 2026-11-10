"""Verify node: deterministic citation checks, then an optional LLM groundedness judge.

Only answers with numbered sources are checked. Answers from tool results alone (for example a
calculation on numbers in the question) have nothing to cite and are marked `skipped`.
"""

import re

from insightflow.agents.graph import AgentNode
from insightflow.agents.models import JudgeVerdict, VerificationResult
from insightflow.agents.observations import has_evidence, number_sources, render_observations
from insightflow.agents.state import AgentState
from insightflow.agents.structured import complete_json
from insightflow.providers.llm import ChatProvider

CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
NO_CONTEXT_ISSUE = "No sources were found."
JUDGE_SYSTEM_PROMPT = (
    "You check whether an answer is supported by observations: numbered sources and tool "
    "results. "
    'Return only a JSON object: {"grounded": true or false, "unsupported_claims": ["..."]}. '
    "Set grounded to true only if the observations support every factual claim in the answer. "
    "A value that a tool calculated counts as supported. "
    "A statement that the observations do not contain the answer is not a factual claim. "
    "List each unsupported claim in a few words."
)


def cited_passages(answer: str) -> set[int]:
    """Return the passage numbers cited as `[n]` or `[n, m]`."""
    return {
        int(number)
        for group in CITATION_PATTERN.findall(answer)
        for number in group.split(",")
    }


def check_citations(answer: str, source_count: int) -> list[str]:
    """Return rule-based problems with an answer's citations; empty means none were found."""
    if not answer.strip():
        return ["The answer is empty."]
    cited = cited_passages(answer)
    if not cited:
        return ["The answer cites no source."]
    missing = sorted(number for number in cited if not 1 <= number <= source_count)
    if missing:
        listed = ", ".join(f"[{number}]" for number in missing)
        return [f"The answer cites sources that do not exist: {listed}."]
    return []


def make_verify_node(judge: ChatProvider | None) -> AgentNode:
    """Bind the optional LLM judge. Without one, only deterministic checks run."""

    async def verify_node(state: AgentState) -> AgentState:
        observations = state.get("observations", [])
        answer = state.get("answer", "")
        sources = number_sources(observations)
        if not sources:
            issues = [] if has_evidence(observations) else [NO_CONTEXT_ISSUE]
            result = VerificationResult(status="skipped", method="deterministic", issues=issues)
            return {"verification": result}

        issues = check_citations(answer, len(sources))
        if issues:
            result = VerificationResult(status="failed", method="deterministic", issues=issues)
            return {"verification": result}
        if judge is None:
            return {"verification": VerificationResult(status="passed", method="deterministic")}

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Observations:\n{render_observations(observations)}\n\n"
                    f"Question: {state['query']}\n\nAnswer: {answer}"
                ),
            },
        ]
        verdict = await complete_json(judge, messages, JudgeVerdict)
        if verdict is None:
            result = VerificationResult(
                status="unchecked",
                method="llm",
                issues=["The judge response could not be read."],
            )
        elif verdict.grounded:
            result = VerificationResult(status="passed", method="llm")
        else:
            claims = [claim.strip() for claim in verdict.unsupported_claims if claim.strip()]
            result = VerificationResult(
                status="failed",
                method="llm",
                issues=claims or ["The judge found unsupported claims."],
            )
        return {"verification": result}

    return verify_node
