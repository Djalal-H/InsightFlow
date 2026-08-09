"""Control-track parser for plain-text sources."""

import re

from insightflow.core.exceptions import DocumentRejectedError
from insightflow.rag.identity import (
    content_checksum,
    create_document_id,
    create_document_version,
    create_element_id,
)
from insightflow.rag.models import DocumentElement, DocumentSource, NormalizedDocument

_SUPPORTED_MIME_TYPES = {"text/plain"}


def _normalize_whitespace(raw: str) -> str:
    """Collapse trailing whitespace and excess blank lines without touching content."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class TxtParser:
    """Parse `.txt` sources into normalized ordered text with no structure."""

    parser_name = "control_txt"
    parser_version = "1.0.0"

    async def parse(self, source: DocumentSource) -> NormalizedDocument:
        if source.mime_type not in _SUPPORTED_MIME_TYPES:
            raise DocumentRejectedError(
                reason="unsupported_format",
                source_identifier=source.source_identifier,
            )

        raw = source.path.read_text(encoding="utf-8", errors="replace")
        text = _normalize_whitespace(raw)

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
            title=None,
            language=None,
            text=text,
            elements=[element],
            parser_name=self.parser_name,
            parser_version=self.parser_version,
            content_checksum=content_checksum(text),
        )