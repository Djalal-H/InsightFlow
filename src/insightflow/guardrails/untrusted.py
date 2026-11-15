"""Screen what a tool or a search returned before it enters the observations."""

from collections.abc import Sequence
from typing import Literal

from insightflow.guardrails.screen import screen
from insightflow.rag.models import RetrievedChunk
from insightflow.security.audit import audit
from insightflow.tools.models import ToolResult

WebAction = Literal["drop", "flag"]


def removed_pages_output(kept: int, removed: int) -> str:
    return (
        f"The search returned {kept + removed} pages. {removed} of them were removed because "
        "they contain text that looks like instructions to the assistant."
    )


def screen_observation(
    result: ToolResult,
    passages: Sequence[RetrievedChunk],
    *,
    web_action: WebAction = "drop",
) -> tuple[ToolResult, list[str]]:
    """Return the result to record and the signals found (`web:...`, `document:...`, `tool:...`).

    Web pages with a signal are removed with `drop` (their raw text also leaves `output`), or
    kept and only reported with `flag`. Passages come from the curated corpus and are only
    reported. Plain tool output is only reported; spotlighting still marks it as data.
    """
    signals: set[str] = set()
    kept = []
    for page in result.web_sources:
        found = screen(f"{page.title}\n{page.content}", untrusted=True)
        signals.update(f"web:{name}" for name in found)
        if found and web_action == "drop":
            continue
        kept.append(page)
    for passage in passages:
        signals.update(f"document:{name}" for name in screen(passage.chunk.content, untrusted=True))
    if result.status == "ok" and not result.web_sources and not passages and result.output:
        signals.update(f"tool:{name}" for name in screen(result.output, untrusted=True))

    removed = len(result.web_sources) - len(kept)
    if removed:
        result = result.model_copy(
            update={"web_sources": kept, "output": removed_pages_output(len(kept), removed)}
        )
    if signals:
        audit(
            "untrusted_content_flagged",
            tool=result.tool,
            signals=sorted(signals),
            removed_pages=removed,
        )
    return result, sorted(signals)
