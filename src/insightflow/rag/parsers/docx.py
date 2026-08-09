"""Control-track parser: flatten a DOCX file into reading-order text."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

from docling.document_converter import DocumentConverter

from insightflow.core.exceptions import DocumentRejectedError
from insightflow.rag.identity import (
    content_checksum,
    create_document_id,
    create_document_version,
    create_element_id,
)
from insightflow.rag.models import DocumentElement, DocumentSource, NormalizedDocument

_SUPPORTED_MIME_TYPES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

ConversionRunner = Callable[[object, Path], Awaitable[object]]


async def _default_conversion_runner(converter: object, source: Path) -> object:
    """Run Docling's blocking convert() off the event loop."""
    return await asyncio.to_thread(
        converter.convert,  # type: ignore[attr-defined]
        source,
        raises_on_error=False,
    )


def _create_docling_converter() -> DocumentConverter:
    """DOCX needs no OCR/table pipeline options, unlike the PDF pipeline."""
    return DocumentConverter()


class DocxParser:
    """Consume Docling's DOCX output and flatten it to ordered text."""

    parser_name = "control_docx_flatten"
    parser_version = "1.0.0"

    def __init__(
        self,
        *,
        converter: object | None = None,
        conversion_runner: ConversionRunner = _default_conversion_runner,
    ) -> None:
        self._converter = converter if converter is not None else _create_docling_converter()
        self._conversion_runner = conversion_runner

    async def parse(self, source: DocumentSource) -> NormalizedDocument:
        if source.mime_type not in _SUPPORTED_MIME_TYPES:
            raise DocumentRejectedError(
                reason="unsupported_format",
                source_identifier=source.source_identifier,
            )

        result = await self._conversion_runner(self._converter, source.path)
        status = getattr(result, "status", None)
        if status is not None and getattr(status, "value", status) == "failure":
            raise DocumentRejectedError(
                reason="conversion_failed",
                source_identifier=source.source_identifier,
            )

        docling_doc = result.document  # type: ignore[attr-defined]
        document_id = create_document_id(source.source_identifier)

        elements: list[DocumentElement] = []
        order = 0
        for text_item in docling_doc.texts:
            text = (text_item.text or "").strip()
            if not text:
                continue
            elements.append(
                DocumentElement(
                    element_id=create_element_id(document_id, self.parser_version, order, text),
                    element_type="paragraph",
                    text=text,
                    page_number=None,
                    order=order,
                )
            )
            order += 1

        if not elements:
            raise DocumentRejectedError(
                reason="empty_document",
                source_identifier=source.source_identifier,
            )

        full_text = "\n\n".join(element.text for element in elements)
        document_version = create_document_version(full_text)

        return NormalizedDocument(
            document_id=document_id,
            document_version=document_version,
            source_name=source.source_name,
            source_identifier=source.source_identifier,
            mime_type=source.mime_type,
            title=None,
            language=None,
            text=full_text,
            elements=elements,
            parser_name=self.parser_name,
            parser_version=self.parser_version,
            content_checksum=content_checksum(full_text),
        )