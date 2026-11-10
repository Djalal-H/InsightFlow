"""Tavily web search through Tavily's hosted MCP server."""

import json
from typing import Any

from insightflow.tools.mcp_client import McpToolClient, http_transport
from insightflow.tools.models import ToolResult, WebSource
from insightflow.tools.registry import ToolRegistration

TAVILY_SERVER_NAME = "tavily"
TAVILY_SEARCH_TOOL = "tavily_search"

MIN_PAGE_CHARS = 200

_LABELS = ("Title: ", "ID: ", "URL: ", "Content: ", "Raw Content: ", "Favicon: ")


def create_tavily_registration(
    url: str,
    api_key: str,
    *,
    max_results: int,
    timeout_seconds: float,
    max_output_chars: int,
) -> ToolRegistration:
    """Register only `tavily_search`, with a fixed result count and no raw page content."""
    headers = {"Authorization": f"Bearer {api_key}"}
    client = McpToolClient(
        lambda: http_transport(url, headers),
        server_name=TAVILY_SERVER_NAME,
        allowed_tools={TAVILY_SEARCH_TOOL},
        timeout_seconds=timeout_seconds,
        max_output_chars=max_output_chars,
    )
    fixed: dict[str, Any] = {
        "max_results": max_results,
        "include_raw_content": False,
        "include_images": False,
    }
    # The pages of one search together stay near the output limit of the other tools.
    page_chars = max(MIN_PAGE_CHARS, max_output_chars // max_results)
    return ToolRegistration(
        client=client,
        fixed_arguments={TAVILY_SEARCH_TOOL: fixed},
        source_parser=lambda result: parse_tavily_sources(result, max_page_chars=page_chars),
    )


def parse_tavily_sources(
    result: ToolResult, *, max_page_chars: int | None = None
) -> list[WebSource]:
    """Read the pages from a Tavily search result.

    The hosted server sends a JSON object with a `results` list, as structured content and as
    text. Structured content is used first, then JSON text, then the older text format with one
    block per page (`Title:`, `URL:`, and `Content:` lines). Pages without an http(s) URL are
    dropped, and page content longer than `max_page_chars` is cut.
    """
    items = (result.structured or {}).get("results")
    if not isinstance(items, list):
        items = _json_results(result.output)
    if isinstance(items, list):
        sources = [
            WebSource(
                title=str(item.get("title") or item.get("url")),
                url=str(item.get("url")),
                content=str(item.get("content") or ""),
            )
            for item in items
            if isinstance(item, dict) and item.get("url")
        ]
    else:
        sources = _parse_text(result.output)
    return [_clip_page(source, max_page_chars) for source in sources if _is_web_url(source.url)]


def _json_results(text: str) -> Any:
    if not text.lstrip().startswith("{"):
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data.get("results") if isinstance(data, dict) else None


def _clip_page(page: WebSource, max_chars: int | None) -> WebSource:
    if max_chars is None or len(page.content) <= max_chars:
        return page
    return page.model_copy(update={"content": page.content[: max_chars - 3].rstrip() + "..."})


def _parse_text(text: str) -> list[WebSource]:
    sources: list[WebSource] = []
    page: dict[str, str] = {}
    field: str | None = None

    def flush() -> None:
        url = page.get("url", "").strip()
        if _is_web_url(url):
            sources.append(
                WebSource(
                    title=page.get("title", "").strip() or url,
                    url=url,
                    content=page.get("content", "").strip(),
                )
            )
        page.clear()

    for line in text.splitlines():
        if line.strip() == "Images:":
            break
        label = next((label for label in _LABELS if line.startswith(label)), None)
        if label is None:
            if field == "content":
                page["content"] = f"{page.get('content', '')}\n{line}"
            continue
        field = label[:-2].lower()
        # A title is the first line of a page when it exists; otherwise a page starts at its URL.
        if (field == "title" and page) or (field == "url" and "url" in page):
            flush()
        if field in ("title", "url", "content"):
            page[field] = line[len(label) :]
    flush()
    return sources


def _is_web_url(url: str) -> bool:
    return url.startswith(("https://", "http://"))
