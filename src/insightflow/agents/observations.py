"""Number the sources of a turn and render observations for prompts.

Document passages and web pages share one numbering per turn, in the order they were first
observed; a passage or page that a later step returns again keeps its first number. The agent,
answer, and verify prompts and the API response all use `number_sources`, so `[n]` in an answer
matches the `citation` of its source.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass

from insightflow.agents.models import SEARCH_TOOL, Observation
from insightflow.rag.models import RetrievedChunk
from insightflow.rag.retrieval import render_context_block
from insightflow.tools.models import ToolSpec, WebSource

TOOL_DESCRIPTION_MAX_CHARS = 300


@dataclass(frozen=True)
class NumberedSource:
    """A citable source: exactly one of `passage` and `page` is set."""

    number: int
    passage: RetrievedChunk | None = None
    page: WebSource | None = None


def passage_key(passage: RetrievedChunk) -> str:
    return f"chunk:{passage.chunk.chunk_id}"


def page_key(page: WebSource) -> str:
    return f"url:{page.url}"


def source_key(source: NumberedSource) -> str:
    """The identity used to number a source once per turn."""
    if source.passage is not None:
        return passage_key(source.passage)
    assert source.page is not None
    return page_key(source.page)


def observation_keys(observation: Observation) -> list[str]:
    """The source identities that one observation returned, in order."""
    return [
        *(passage_key(passage) for passage in observation.passages),
        *(page_key(page) for page in observation.result.web_sources),
    ]


def number_sources(observations: Sequence[Observation]) -> list[NumberedSource]:
    """Return the turn's distinct sources with their citation numbers."""
    sources: list[NumberedSource] = []
    seen: set[str] = set()
    for observation in observations:
        for passage in observation.passages:
            key = passage_key(passage)
            if key not in seen:
                seen.add(key)
                sources.append(NumberedSource(number=len(sources) + 1, passage=passage))
        for page in observation.result.web_sources:
            key = page_key(page)
            if key not in seen:
                seen.add(key)
                sources.append(NumberedSource(number=len(sources) + 1, page=page))
    return sources


def render_observations(observations: Sequence[Observation]) -> str:
    """Render each step with its call, outcome, and newly numbered sources."""
    if not observations:
        return "No observations yet."
    numbers = {source_key(source): source.number for source in number_sources(observations)}
    shown: set[str] = set()
    blocks: list[str] = []
    for step, observation in enumerate(observations, start=1):
        result = observation.result
        arguments = json.dumps(result.arguments, ensure_ascii=False, sort_keys=True)
        header = f"Step {step}: {result.tool} {arguments}"
        if result.status == "error":
            blocks.append(f"{header} failed ({result.error}): {result.output}")
            continue

        items = observation_keys(observation)
        if not items:
            if _is_search(observation):
                blocks.append(f"{header} found no sources.")
            else:
                blocks.append(f"{header} returned: {result.output}")
            continue

        new = [key for key in items if key not in shown]
        repeated = sorted({numbers[key] for key in items if key in shown})
        lines = [f"{header} returned sources:"]
        lines.extend(_render_source(key, numbers[key], observation) for key in new)
        if repeated:
            lines.append("Already shown: " + ", ".join(f"[{number}]" for number in repeated))
        shown.update(new)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def render_tool_catalog(specs: Sequence[ToolSpec]) -> str:
    """List tools with clipped descriptions and their input schemas."""
    lines = []
    for spec in specs:
        description = spec.description.strip().replace("\n", " ")
        if len(description) > TOOL_DESCRIPTION_MAX_CHARS:
            description = description[: TOOL_DESCRIPTION_MAX_CHARS - 3] + "..."
        schema = json.dumps(spec.input_schema, sort_keys=True)
        lines.append(f"- {spec.name}: {description}\n  input schema: {schema}")
    return "\n".join(lines)


def has_evidence(observations: Sequence[Observation]) -> bool:
    """True when some step gave the answer something to use.

    That is a source, a tool output, or an error that the tool itself reported (for example
    "division by zero"), which the answer can explain. Timeouts, unavailable servers, unknown
    tools, invalid arguments, and empty searches are not evidence.
    """
    return any(
        observation.passages
        or observation.result.web_sources
        or (
            not _is_search(observation)
            and (observation.result.status == "ok" or observation.result.error == "tool_failed")
        )
        for observation in observations
    )


def _is_search(observation: Observation) -> bool:
    return observation.result.tool == SEARCH_TOOL


def _render_source(key: str, number: int, observation: Observation) -> str:
    passage = next((p for p in observation.passages if passage_key(p) == key), None)
    if passage is not None:
        return render_context_block(number, passage)
    page = next(p for p in observation.result.web_sources if page_key(p) == key)
    return f"[{number}] {page.title} ({page.url})\n{page.content}"
