"""Control-track parser for Markdown sources: flattened to text, title kept."""

import re

import markdown as markdown_lib

from insightflow.core.exceptions import DocumentRejectedError
from insightflow.rag.identity import (
    content_checksum,
    create_document_id,
    create_document_version,
    create_element_id,
)
from insightflow.rag.models import DocumentElement, DocumentSource, NormalizedDocument

_SUPPORTED_MIME_TYPES = {"text/markdown"}
_TAG_RE = re.compile(r"<[^>]+>")
_TITLE_RE = re.compile(r"^#\s+(.+?)(?:\s*\{[^}]*\})?\s*$", re.MULTILINE)
_BLANK_RUN_RE = re.compile(r"\n{3,}")


def _flatten(raw_markdown: str) -> str:
    """Render Markdown to HTML, then strip tags to get plain reading-order text."""
    html = markdown_lib.markdown(raw_markdown)
    text = _TAG_RE.sub("", html)
    text = _BLANK_RUN_RE.sub("\n\n", text)
    return text.strip()


def _extract_title(raw_markdown: str) -> str | None:
    """Pull the first top-level heading as the essential title metadata."""
    match = _TITLE_RE.search(raw_markdown)
    return match.group(1).strip() if match else None


class MarkdownParser:
    """Parse `.md` sources into flattened text, preserving only the title."""

    parser_name = "control_markdown"
    parser_version = "1.0.0"

    async def parse(self, source: DocumentSource) -> NormalizedDocument:
        if source.mime_type not in _SUPPORTED_MIME_TYPES:
            raise DocumentRejectedError(
                reason="unsupported_format",
                source_identifier=source.source_identifier,
            )

        raw = source.path.read_text(encoding="utf-8", errors="replace")
        title = _extract_title(raw)
        text = _flatten(raw)

        if not text:
            raise DocumentRejectedError(
                reason="empty_document",
                source_identifier=source.source_identifier,
            )

        document_id = create_document_id(source.source_identifier)
        document_version = create_document_version(text)

        element = DocumentElement(
            element_id=create_element_id(document_id, self.parser_version, 0, text),
            element_type="paragraph",
            text=text,
            order=0,
        )

        return NormalizedDocument(
            document_id=document_id,
            document_version=document_version,
            source_name=source.source_name,
            source_identifier=source.source_identifier,
            mime_type=source.mime_type,
            title=title,
            language=None,
            text=text,
            elements=[element],
            parser_name=self.parser_name,
            parser_version=self.parser_version,
            content_checksum=content_checksum(text),
        )