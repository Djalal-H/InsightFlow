# src/insightflow/rag/parsers/docx.py
"""Control-track parser: flatten a DOCX file into reading-order text."""

import asyncio

from docling.document_converter import DocumentConverter

from insightflow.rag.identity import (
    content_checksum,
    create_document_id,
    create_document_version,
    create_element_id,
)
from insightflow.rag.models import DocumentElement, DocumentSource, NormalizedDocument


class DocxParser:
    """Consume Docling's DOCX output and flatten it to ordered text."""

    parser_name = "control_docx_flatten"
    parser_version = "1.0.0"

    def __init__(self) -> None:
        self._converter = DocumentConverter()

    async def parse(self, source: DocumentSource) -> NormalizedDocument:
        result = await asyncio.to_thread(self._converter.convert, str(source.path))
        docling_doc = result.document

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
                    page_number=None,  # DOCX has no reliable page concept
                    order=order,
                )
            )
            order += 1

        if not elements:
            raise ValueError(f"{source.source_identifier}: no extractable text found")

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
