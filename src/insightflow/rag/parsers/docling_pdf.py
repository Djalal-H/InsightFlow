"""Structural parser for digital PDFs converted by Docling."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast
from unicodedata import normalize

from pydantic import JsonValue
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from insightflow.core.exceptions import DocumentRejectedError, DocumentRejectionReason
from insightflow.rag.identity import (
    content_checksum,
    create_document_id,
    create_document_version,
    create_element_id,
)
from insightflow.rag.models import DocumentElement, DocumentSource, ElementType, NormalizedDocument

PDF_MIME_TYPES = frozenset({"application/pdf", "application/x-pdf"})


class _Converter(Protocol):
    """Small duck-typed boundary around Docling's converter."""

    def convert(self, source: Path, *, raises_on_error: bool) -> object: ...


async def _convert_in_thread(converter: _Converter, source: Path) -> object:
    """Keep synchronous PDF conversion off the application's event loop."""
    return await asyncio.to_thread(converter.convert, source, raises_on_error=False)


@dataclass(frozen=True)
class PdfProfile:
    """Facts needed to enforce the digital-PDF ingestion policy."""

    page_count: int
    encrypted: bool
    has_images: bool


@dataclass(frozen=True)
class _ElementDraft:
    """One mapped element before deterministic IDs and offsets are assigned."""

    element_type: ElementType
    text: str
    heading_path: list[str]
    page_numbers: list[int]
    metadata: dict[str, JsonValue] = field(default_factory=dict)


def inspect_pdf(path: Path) -> PdfProfile:
    """Inspect encryption and raster content without performing OCR."""
    try:
        reader = PdfReader(path)
        if reader.is_encrypted:
            return PdfProfile(page_count=0, encrypted=True, has_images=False)
        pages = list(reader.pages)
        return PdfProfile(
            page_count=len(pages),
            encrypted=False,
            has_images=any(len(page.images) > 0 for page in pages),
        )
    except (OSError, PdfReadError, ValueError) as exc:
        raise ValueError("PDF inspection failed") from exc


def _create_docling_converter() -> _Converter:
    """Create a PDF-only Docling converter with OCR explicitly disabled."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
    return cast(
        _Converter,
        DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=options),
            },
        ),
    )


class DoclingPdfParser:
    """Map Docling's PDF representation into InsightFlow domain elements."""

    parser_name = "docling_pdf"
    parser_version = "1"

    def __init__(
        self,
        *,
        converter: _Converter | None = None,
        pdf_inspector: Callable[[Path], PdfProfile] = inspect_pdf,
        conversion_runner: Callable[[_Converter, Path], Awaitable[object]] = _convert_in_thread,
    ) -> None:
        self._converter = converter
        self._pdf_inspector = pdf_inspector
        self._conversion_runner = conversion_runner

    async def parse(self, source: DocumentSource) -> NormalizedDocument:
        """Convert one digital PDF into ordered provider-neutral elements."""
        self._validate_source_format(source)
        profile = self._inspect_source(source)
        if profile.encrypted:
            self._reject(source, "encrypted_pdf")

        converter = self._converter or _create_docling_converter()
        try:
            result = await self._conversion_runner(converter, source.path)
        except Exception as exc:
            raise DocumentRejectedError(
                reason="conversion_failed",
                source_identifier=source.source_identifier,
            ) from exc

        if not self._conversion_succeeded(result):
            self._reject(source, "conversion_failed")

        document = getattr(result, "document", None)
        if document is None:
            self._reject(source, "conversion_failed")

        drafts = self._map_document(document)
        if not drafts:
            reason: DocumentRejectionReason = (
                "scanned_pdf" if profile.has_images else "textless_pdf"
            )
            self._reject(source, reason)

        return self._assemble_document(source, profile, drafts)

    @staticmethod
    def _validate_source_format(source: DocumentSource) -> None:
        if source.mime_type.lower() not in PDF_MIME_TYPES or source.path.suffix.lower() != ".pdf":
            raise DocumentRejectedError(
                reason="unsupported_format",
                source_identifier=source.source_identifier,
            )
        if not source.path.is_file():
            raise DocumentRejectedError(
                reason="conversion_failed",
                source_identifier=source.source_identifier,
            )

    def _inspect_source(self, source: DocumentSource) -> PdfProfile:
        try:
            return self._pdf_inspector(source.path)
        except Exception as exc:
            raise DocumentRejectedError(
                reason="conversion_failed",
                source_identifier=source.source_identifier,
            ) from exc

    @staticmethod
    def _conversion_succeeded(result: object) -> bool:
        status = getattr(result, "status", None)
        if status is None:
            return True
        value = str(getattr(status, "value", status)).lower()
        return value == "success"

    def _map_document(self, document: object) -> list[_ElementDraft]:
        iterate_items = getattr(document, "iterate_items", None)
        if not callable(iterate_items):
            return []

        drafts: list[_ElementDraft] = []
        heading_stack: list[str] = []
        items = cast(Iterable[tuple[object, int]], iterate_items())
        for item, tree_level in items:
            label = self._label_value(item)
            element_type = self._element_type(label)
            if element_type is None:
                continue

            text = self._item_text(item, document, element_type)
            if not self._has_meaningful_text(text):
                continue

            metadata: dict[str, JsonValue] = {}
            if element_type == "heading":
                heading_level = self._heading_level(item, tree_level)
                heading_stack = heading_stack[: heading_level - 1]
                heading_stack.append(text)
                metadata["heading_level"] = heading_level
            elif element_type == "list_item":
                metadata.update(self._list_metadata(item, tree_level))
            elif element_type == "code":
                language = getattr(item, "language", None)
                if isinstance(language, str) and language.strip():
                    metadata["language"] = language.strip()
            elif element_type == "table":
                metadata.update(self._table_metadata(item))
            elif element_type == "other":
                metadata["source_role"] = label or type(item).__name__.lower()

            page_numbers = self._page_numbers(item)
            if len(page_numbers) > 1:
                metadata["page_numbers"] = cast(JsonValue, page_numbers)
            drafts.append(
                _ElementDraft(
                    element_type=element_type,
                    text=text,
                    heading_path=list(heading_stack),
                    page_numbers=page_numbers,
                    metadata=metadata,
                )
            )
        return drafts

    @staticmethod
    def _label_value(item: object) -> str:
        label = getattr(item, "label", "")
        return str(getattr(label, "value", label)).lower()

    @staticmethod
    def _element_type(label: str) -> ElementType | None:
        mapping: dict[str, ElementType] = {
            "title": "title",
            "section_header": "heading",
            "text": "paragraph",
            "paragraph": "paragraph",
            "list_item": "list_item",
            "code": "code",
            "table": "table",
            "caption": "caption",
        }
        if label in mapping:
            return mapping[label]
        if label in {"picture", "page", "document_index"}:
            return None
        return "other" if label else None

    @classmethod
    def _item_text(cls, item: object, document: object, element_type: ElementType) -> str:
        if element_type == "table":
            exporter = getattr(item, "export_to_markdown", None)
            if not callable(exporter):
                return ""
            try:
                value = exporter(doc=document)
            except TypeError:
                value = exporter()
        else:
            value = getattr(item, "text", "")
        if not isinstance(value, str):
            return ""
        normalized = normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))
        return normalized.rstrip("\n") if element_type == "code" else normalized.strip()

    @staticmethod
    def _heading_level(item: object, tree_level: int) -> int:
        explicit = getattr(item, "level", None)
        if isinstance(explicit, int) and explicit >= 1:
            return explicit
        return max(1, tree_level)

    @staticmethod
    def _list_metadata(item: object, tree_level: int) -> dict[str, JsonValue]:
        enumerated = bool(getattr(item, "enumerated", False))
        metadata: dict[str, JsonValue] = {
            "list_kind": "ordered" if enumerated else "unordered",
            "list_depth": max(1, tree_level),
        }
        marker = getattr(item, "marker", None)
        if enumerated and isinstance(marker, str):
            ordinal = marker.rstrip(".) ")
            if ordinal.isdigit():
                metadata["ordinal"] = int(ordinal)
        return metadata

    @staticmethod
    def _table_metadata(item: object) -> dict[str, JsonValue]:
        data = getattr(item, "data", None)
        metadata: dict[str, JsonValue] = {}
        for source_name, target_name in (("num_rows", "row_count"), ("num_cols", "column_count")):
            value = getattr(data, source_name, None)
            if isinstance(value, int) and value >= 0:
                metadata[target_name] = value
        return metadata

    @staticmethod
    def _page_numbers(item: object) -> list[int]:
        provenance = cast(Sequence[object], getattr(item, "prov", ()) or ())
        pages = {
            page
            for entry in provenance
            if isinstance((page := getattr(entry, "page_no", None)), int) and page >= 1
        }
        return sorted(pages)

    @staticmethod
    def _has_meaningful_text(text: str) -> bool:
        return any(character.isalnum() for character in text)

    def _assemble_document(
        self,
        source: DocumentSource,
        profile: PdfProfile,
        drafts: list[_ElementDraft],
    ) -> NormalizedDocument:
        document_id = create_document_id(source.source_identifier)
        parts: list[str] = []
        elements: list[DocumentElement] = []
        cursor = 0
        title: str | None = None

        for order, draft in enumerate(drafts):
            if parts:
                cursor += 2
            char_start = cursor
            char_end = char_start + len(draft.text)
            element = DocumentElement(
                element_id=create_element_id(
                    document_id,
                    self.parser_version,
                    order,
                    draft.text,
                ),
                element_type=draft.element_type,
                text=draft.text,
                heading_path=draft.heading_path,
                page_number=draft.page_numbers[0] if draft.page_numbers else None,
                order=order,
                char_start=char_start,
                char_end=char_end,
                metadata=draft.metadata,
            )
            if title is None and draft.element_type == "title":
                title = draft.text
            parts.append(draft.text)
            elements.append(element)
            cursor = char_end

        text = "\n\n".join(parts)
        metadata = dict(source.metadata)
        metadata["page_count"] = profile.page_count
        return NormalizedDocument(
            document_id=document_id,
            document_version=create_document_version(text),
            source_name=source.source_name,
            source_identifier=source.source_identifier,
            mime_type=source.mime_type,
            title=title,
            text=text,
            elements=elements,
            metadata=metadata,
            parser_name=self.parser_name,
            parser_version=self.parser_version,
            content_checksum=content_checksum(text),
        )

    @staticmethod
    def _reject(source: DocumentSource, reason: DocumentRejectionReason) -> None:
        raise DocumentRejectedError(
            reason=reason,
            source_identifier=source.source_identifier,
        )
